#!/usr/bin/env python3
"""Restore only this project's Raspberry Pi host route in mirrored WSL."""

import argparse
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


MTU = 1400
ADVMSS = 1360
DEFAULT_CONFIG = Path(__file__).with_name("network.env")


def load_config(path):
    """Read literal KEY=value data; never execute shell statements."""
    config = {}
    for number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError(f"配置第 {number} 行不是 KEY=value")
        if key in config:
            raise ValueError(f"配置字段重复：{key}")
        config[key] = value.strip()
    for key in ("PI_ZT_IP", "WINDOWS_ZT_IP", "WSL_ZT_IP"):
        try:
            address = ipaddress.IPv4Address(config[key])
        except (KeyError, ipaddress.AddressValueError) as exc:
            raise ValueError(f"{key} 必须是明确的 IPv4 地址") from exc
        if address.is_unspecified or address.is_multicast or address.is_loopback or int(address) == 0xFFFFFFFF:
            raise ValueError(f"{key} 不能是未指定、组播、回环或广播地址")
        config[key] = str(address)
    if config["PI_ZT_IP"] in (config["WINDOWS_ZT_IP"], config["WSL_ZT_IP"]):
        raise ValueError("树莓派地址不能与本机地址相同")
    if config["WSL_ZT_IP"] != config["WINDOWS_ZT_IP"]:
        raise ValueError("此脚本用于 mirrored 模式，WSL_ZT_IP 须与 WINDOWS_ZT_IP 相同")
    port = config.get("CHAT_PORT", "")
    if not re.fullmatch(r"[0-9]{1,5}", port) or not 1 <= int(port) <= 65535:
        raise ValueError("CHAT_PORT 必须是 1 到 65535 的整数")
    config["CHAT_PORT"] = str(int(port))
    return config


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, errors="replace", timeout=20)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or f"退出码 {result.returncode}"
        raise RuntimeError(f"命令执行失败：{detail}")
    return result.stdout.strip()


def ip_json(ip, *args):
    result = json.loads(command([ip, "-j", "-4", *args]))
    if not isinstance(result, list):
        raise RuntimeError("ip -j 返回了意外格式")
    return result


def exact_routes(ip, destination):
    routes = ip_json(ip, "route", "show", "table", "all", "exact", destination)
    return [route for route in routes if str(ipaddress.ip_network(route.get("dst", "0.0.0.0/0"))) == destination]


def equivalent(route, interface, source):
    metrics = route.get("metrics", {})
    if isinstance(metrics, list):
        metrics = {key: value for entry in metrics for key, value in entry.items()}
    return (
        route.get("dev") == interface
        and route.get("prefsrc") == source
        and route.get("table", "main") in ("main", 254)
        and route.get("type", "unicast") == "unicast"
        and not route.get("gateway") and not route.get("nexthops")
        and metrics.get("mtu") == MTU and metrics.get("advmss") == ADVMSS
    )


def prepare(config):
    ip = shutil.which("ip")
    if not ip:
        raise RuntimeError("未找到 ip 命令，请在真实 WSL Ubuntu 终端运行")
    source = config["WINDOWS_ZT_IP"]
    interfaces = {
        entry["ifname"] for entry in ip_json(ip, "addr", "show")
        if any(address.get("local") == source for address in entry.get("addr_info", []))
    }
    if len(interfaces) != 1:
        raise RuntimeError(f"Windows ZeroTier 地址 {source} 未出现在唯一的 WSL 网卡上，请检查 mirrored 网络和 ZeroTier")
    interface = interfaces.pop()
    destination = config["PI_ZT_IP"] + "/32"
    routes = exact_routes(ip, destination)
    if routes and (len(routes) != 1 or not equivalent(routes[0], interface, source)):
        raise RuntimeError(f"{destination} 已有冲突路由，未作任何修改：{json.dumps(routes, ensure_ascii=False)}")
    if not routes:
        distro = os.environ.get("WSL_DISTRO_NAME") or config.get("WSL_DISTRO_NAME", "")
        if not distro or not distro.isprintable() or distro != distro.strip() or distro.startswith("-"):
            raise RuntimeError("缺少有效 WSL_DISTRO_NAME，请在目标 WSL 发行版终端运行")
        wsl = shutil.which("wsl.exe")
        if not wsl:
            fallback = Path("/mnt/c/Windows/System32/wsl.exe")
            if fallback.is_file():
                wsl = str(fallback)
        if not wsl:
            raise RuntimeError("未找到 Windows wsl.exe；请检查 WSL interop 和 Windows 系统盘挂载")
        try:
            command([
                wsl, "--distribution", distro, "--user", "root", "--exec", ip,
                "-4", "route", "add", destination, "dev", interface, "src", source,
                "mtu", str(MTU), "advmss", str(ADVMSS),
            ])
        except RuntimeError:
            # An equivalent concurrent add is safe; never replace or remove a route.
            routes = exact_routes(ip, destination)
            if len(routes) != 1 or not equivalent(routes[0], interface, source):
                raise
        routes = exact_routes(ip, destination)
        if len(routes) != 1 or not equivalent(routes[0], interface, source):
            raise RuntimeError("添加后未找到预期的 /32 路由，请检查 WSL 网络状态")
        print(f"已准备树莓派专用路由：{destination}，MTU {MTU}，MSS {ADVMSS}", file=sys.stderr)
    else:
        print(f"树莓派专用路由已就绪：{destination}，无需修改", file=sys.stderr)
    print("当前有效路由：\n" + command([ip, "-4", "route", "get", config["PI_ZT_IP"]]), file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="仅为本项目树莓派恢复 WSL /32 路由和 TCP MSS；不修改网卡 MTU或默认路由")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-config", action="store_true", help="仅校验配置，不读取或修改系统网络")
    modes.add_argument("--server-args", action="store_true", help="准备网络后向标准输出写入已校验的 host 和 port，供启动脚本读取")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.check_config:
            print(f"配置有效：WSL {config['WSL_ZT_IP']}，树莓派 {config['PI_ZT_IP']}，TCP {config['CHAT_PORT']}")
            return
        prepare(config)
        if args.server_args:
            print(config["WSL_ZT_IP"])
            print(config["CHAT_PORT"])
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f"网络准备失败：{exc}\n")


if __name__ == "__main__":
    main()
