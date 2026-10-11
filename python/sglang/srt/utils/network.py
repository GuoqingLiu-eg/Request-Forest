from __future__ import annotations

import ipaddress
import logging
import os
import socket
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import psutil
import zmq

logger = logging.getLogger(__name__)


def get_open_port() -> int:
    from sglang.srt.environ import envs

    port = envs.SGLANG_PORT.get()
    if port is not None:
        while True:
            if is_port_available(port):
                return port
            logger.info("Port %d is already in use, trying port %d", port, port + 1)
            port += 1
    sock = try_bind_socket()
    port = sock.getsockname()[1]
    sock.close()
    return port


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def find_process_using_port(port: int) -> Optional[psutil.Process]:
    for conn in psutil.net_connections(kind="inet"):
        if conn.laddr.port == port:
            try:
                return psutil.Process(conn.pid)
            except psutil.NoSuchProcess:

                pass

    return None


MAX_VALID_PORT = 65535


def wait_port_available(
    port: int,
    port_name: str,
    timeout_s: Optional[int] = None,
    raise_exception: bool = True,
) -> bool:
    if timeout_s is None:


        timeout_s = int(os.environ.get("SGLANG_WAIT_PORT_TIMEOUT", "30"))
    if port < 0 or port > MAX_VALID_PORT:
        raise ValueError(
            f"{port_name} has invalid port number {port}. "
            f"Valid TCP port range is 0-{MAX_VALID_PORT}."
        )

    error_message = f"{port_name} at {port} is not available"
    for i in range(timeout_s):
        if is_port_available(port):
            return True

        if i > 10 and i % 5 == 0:
            process = find_process_using_port(port)
            if process is None:
                logger.warning(
                    f"The port {port} is in use, but we could not find the process that uses it."
                )
            else:
                pid = process.pid
                error_message = f"{port_name} is used by a process already. {process.name()=}' {process.cmdline()=} {process.status()=} {pid=}"
                logger.info(
                    f"port {port} is in use. Waiting for {i} seconds for {port_name} to be available. {error_message}"
                )
        time.sleep(1)

    if raise_exception:
        raise ValueError(
            f"{port_name} at {port} is not available in {timeout_s} seconds. {error_message}"
        )
    return False


def _get_addrinfos_for_bind(host=None, port=0):

    try:
        infos = socket.getaddrinfo(
            host,
            port,
            socket.AF_UNSPEC,
            socket.SOCK_STREAM,
            0,
            socket.AI_ADDRCONFIG | socket.AI_PASSIVE,
        )
        deduped = []
        seen_families = set()
        for info in infos:
            if info[0] not in seen_families:
                seen_families.add(info[0])
                deduped.append(info)


        deduped.sort(key=lambda x: (x[0] != socket.AF_INET,))
        return deduped
    except socket.gaierror:
        fallback_host = "0.0.0.0" if host is None else host
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (fallback_host, port))]


def try_bind_socket(host=None, port=0, *, reuse_addr=True, listen=False):

    for family, socktype, proto, _, sockaddr in _get_addrinfos_for_bind(host, port):
        sock = socket.socket(family, socktype, proto)
        try:
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            if reuse_addr:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(sockaddr)
            if listen:
                sock.listen(1)
            return sock
        except (OSError, OverflowError):
            sock.close()
    raise OSError(f"Could not bind port {port} on any configured address family")


def is_port_available(port):

    try:
        for family, socktype, proto, _, sockaddr in _get_addrinfos_for_bind(port=port):
            sock = socket.socket(family, socktype, proto)
            try:
                if family == socket.AF_INET6:
                    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(sockaddr)
            finally:
                sock.close()
        return True
    except (OSError, OverflowError):
        return False


def get_free_port():
    sock = try_bind_socket()
    port = sock.getsockname()[1]
    sock.close()
    return port


def bind_port(port):

    return try_bind_socket(port=port, listen=True)


def get_zmq_socket_on_host(
    context: zmq.Context,
    socket_type: zmq.SocketType,
    host: Optional[str] = None,
) -> Tuple[int, zmq.Socket]:

    socket = context.socket(socket_type)
    config_socket(socket, socket_type)
    if host is None:
        host = "127.0.0.1"
    if is_valid_ipv6_address(host):
        socket.setsockopt(zmq.IPV6, 1)
        bind_host = f"tcp://[{host}]"
    else:
        bind_host = f"tcp://{host}"
    port = socket.bind_to_random_port(bind_host)
    return port, socket


def config_socket(socket, socket_type: zmq.SocketType):
    try:
        mem = psutil.virtual_memory()
        total_mem = mem.total / 1024**3
        available_mem = mem.available / 1024**3
    except Exception as e:
        logger.warning(
            "psutil.virtual_memory() failed (%s); using default ZMQ buffer size", e
        )
        total_mem = available_mem = 0
    if total_mem > 32 and available_mem > 16:
        buf_size = int(0.5 * 1024**3)
    else:
        buf_size = -1

    def set_send_opt():
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    def set_recv_opt():
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type == zmq.PUSH:
        set_send_opt()
    elif socket_type == zmq.PULL:
        set_recv_opt()
    elif socket_type in [zmq.DEALER, zmq.REQ, zmq.REP, zmq.PAIR]:
        set_send_opt()
        set_recv_opt()
    else:
        raise ValueError(f"Unsupported socket type: {socket_type}")


def get_local_ip_by_nic(interface: str = None) -> Optional[str]:
    if not (interface := interface or os.environ.get("SGLANG_LOCAL_IP_NIC", None)):
        return None
    try:
        import netifaces
    except ImportError as e:
        raise ImportError(
            "Environment variable SGLANG_LOCAL_IP_NIC requires package netifaces, please install it through 'pip install netifaces'"
        ) from e

    try:
        addresses = netifaces.ifaddresses(interface)
        if netifaces.AF_INET in addresses:
            for addr_info in addresses[netifaces.AF_INET]:
                ip = addr_info.get("addr")
                if ip and ip != "127.0.0.1" and ip != "0.0.0.0":
                    return ip
        if netifaces.AF_INET6 in addresses:
            for addr_info in addresses[netifaces.AF_INET6]:
                ip = addr_info.get("addr")
                if ip and not ip.startswith("fe80::") and ip != "::1":
                    return ip.split("%")[0]
    except (ValueError, OSError) as e:
        logger.warning(
            f"{e} Can not get local ip from NIC. Please verify whether SGLANG_LOCAL_IP_NIC is set correctly."
        )
    return None


def get_local_ip_by_remote() -> Optional[str]:


    for dns_host, dns_port in [("8.8.8.8", 80), ("2001:4860:4860::8888", 80)]:
        try:
            family, socktype, proto, _, sockaddr = socket.getaddrinfo(
                dns_host,
                dns_port,
                socket.AF_UNSPEC,
                socket.SOCK_DGRAM,
                0,
                socket.AI_ADDRCONFIG,
            )[0]
            with socket.socket(family, socktype, proto) as s:
                s.connect(sockaddr)
                return s.getsockname()[0]
        except (socket.gaierror, OSError):
            continue


    try:
        hostname = socket.gethostname()
        ip = socket.getaddrinfo(
            hostname, None, socket.AF_UNSPEC, 0, 0, socket.AI_ADDRCONFIG
        )[0][4][0]
        if ip and ip not in ("127.0.0.1", "0.0.0.0", "::1"):
            return ip
    except Exception:
        pass

    logger.warning("Can not get local ip by remote")
    return None


def get_local_ip_auto(fallback: str = None) -> str:


    host_ip = os.getenv("SGLANG_HOST_IP", "") or os.getenv("HOST_IP", "")
    if host_ip:
        return host_ip
    logger.debug("get_ip failed")

    if ip := get_local_ip_by_nic():
        return ip
    logger.debug("get_local_ip_by_nic failed")

    if ip := get_local_ip_by_remote():
        return ip
    logger.debug("get_local_ip_by_remote failed")
    if fallback:
        return fallback
    raise ValueError("Can not get local ip")


def get_zmq_socket(
    context: zmq.Context,
    socket_type: zmq.SocketType,
    endpoint: Optional[str] = None,
    bind: bool = True,
) -> Union[zmq.Socket, Tuple[int, zmq.Socket]]:

    socket = context.socket(socket_type)

    if endpoint is None:

        config_socket(socket, socket_type)
        port = socket.bind_to_random_port("tcp://*")
        return port, socket
    else:
        if is_zmq_endpoint_ipv6(endpoint):
            socket.setsockopt(zmq.IPV6, 1)

        config_socket(socket, socket_type)

        if bind:
            socket.bind(endpoint)
        else:
            socket.connect(endpoint)

        return socket


def is_zmq_endpoint_ipv6(endpoint: str) -> bool:

    prefix = "tcp://["
    if not endpoint.startswith(prefix):
        return False
    end = endpoint.find("]", len(prefix))
    if end == -1:
        return False
    return is_valid_ipv6_address(endpoint[len(prefix) : end])


def _is_ipv6(host: str) -> bool:

    try:
        ipaddress.IPv6Address(host)
        return True
    except ValueError:
        return False


def _wrap(host: str) -> str:

    return f"[{host}]" if _is_ipv6(host) else host


def _parse_port(s: str) -> int:
    try:
        port = int(s)
    except ValueError:
        raise ValueError(f"Invalid port number: {s!r}")
    if not (0 <= port <= 65535):
        raise ValueError(f"Port out of range (0-65535): {port}")
    return port


@dataclass(frozen=True)
class NetworkAddress:
    host: str
    port: int

    def __post_init__(self):

        if self.host.startswith("[") and self.host.endswith("]"):
            object.__setattr__(self, "host", self.host[1:-1])

    @property
    def is_ipv6(self) -> bool:
        return _is_ipv6(self.host)

    @property
    def family(self) -> socket.AddressFamily:
        return socket.AF_INET6 if self.is_ipv6 else socket.AF_INET

    def to_url(self, scheme: str = "http") -> str:

        return f"{scheme}://{_wrap(self.host)}:{self.port}"

    def to_tcp(self) -> str:

        return self.to_url("tcp")

    def to_host_port_str(self) -> str:

        return f"{_wrap(self.host)}:{self.port}"

    @staticmethod
    def resolve_host(host: str) -> str:

        try:
            ipaddress.ip_address(host)
            return host
        except ValueError:
            pass
        try:
            return socket.getaddrinfo(
                host, None, socket.AF_UNSPEC, 0, 0, socket.AI_ADDRCONFIG
            )[0][4][0]
        except socket.gaierror as e:
            raise ValueError(f"Cannot resolve host {host!r}: {e}") from e

    def resolved(self) -> NetworkAddress:

        ip = self.resolve_host(self.host)
        return self if ip == self.host else NetworkAddress(ip, self.port)

    def to_bind_tuple(self) -> Tuple[str, int]:

        return (self.host, self.port)

    @staticmethod
    def parse(addr: str) -> NetworkAddress:

        if not addr:
            raise ValueError("Empty address string")


        if addr.startswith("["):
            close = addr.find("]")
            if close == -1:
                raise ValueError(f"Missing closing bracket in IPv6 address: {addr!r}")
            host = addr[1:close]
            if not _is_ipv6(host):
                raise ValueError(f"Invalid IPv6 address inside brackets: {host!r}")
            rest = addr[close + 1 :]
            if not rest.startswith(":") or len(rest) < 2:
                raise ValueError(
                    f"Expected ':port' after closing bracket, got: {rest!r}"
                )
            return NetworkAddress(host, _parse_port(rest[1:]))


        if ":" not in addr:
            raise ValueError(f"Missing port in address (expected host:port): {addr!r}")
        host, port_str = addr.rsplit(":", 1)
        if not host:
            raise ValueError(f"Empty host in address: {addr!r}")

        if ":" in host and _is_ipv6(host):
            raise ValueError(
                f"Bare IPv6 address without brackets is ambiguous: {addr!r}. "
                f"Use [{host}]:{port_str} instead."
            )
        return NetworkAddress(host, _parse_port(port_str))

    def __str__(self) -> str:
        return self.to_host_port_str()

    def __repr__(self) -> str:
        return f"NetworkAddress({self.host!r}, {self.port})"


def resolve_base_url(base_url: str, host: str, port: int) -> str:

    if base_url:
        return base_url
    return NetworkAddress(host, port).to_url()


def resolve_host_port(base_url: str, host: str, port: int) -> str:

    if base_url:
        return base_url
    return NetworkAddress(host, port).to_host_port_str()
