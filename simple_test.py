#!/usr/bin/env python3
"""
简单的单步测试脚本
"""

import asyncio
import sys
import os

sys.path.append(os.path.abspath('.'))
from app.services.order_executor import OrderExecutor

async def test_balance():
    """测试余额查询"""
    executor = OrderExecutor(is_testnet=True)
    try:
        balance = await executor.get_balance('USDT')
        print(f"余额: {balance:.2f} USDT")
        return balance > 0
    finally:
        await executor.close_connections()

async def test_leverage():
    """测试设置杠杆"""
    executor = OrderExecutor(is_testnet=True)
    try:
        success = await executor.set_leverage("BNB/USDT", 5)
        print(f"设置杠杆: {'成功' if success else '失败'}")
        return success
    finally:
        await executor.close_connections()

async def test_order():
    """测试下单"""
    executor = OrderExecutor(is_testnet=True)
    try:
        # 先设置杠杆
        await executor.set_leverage("BNB/USDT", 5)
        
        # 下单 (21 USDT名义价值)
        order = await executor.create_market_order_by_notional("BNB/USDT", "buy", 21.0)
        if order:
            print(f"下单成功: {order.get('id')}")
            print(f"成交数量: {order.get('filled')}")
            
            # 立即平仓
            filled = float(order.get('filled', 0))
            if filled > 0:
                close_order = await executor.close_market_position("BNB/USDT", "LONG", filled)
                print(f"平仓: {'成功' if close_order else '失败'}")
            
            return True
        else:
            print("下单失败")
            return False
    finally:
        await executor.close_connections()

if __name__ == "__main__":
    print("选择测试:")
    print("1. 测试余额")
    print("2. 测试杠杆")
    print("3. 测试下单+平仓")
    
    choice = input("输入选择 (1-3): ").strip()
    
    if choice == "1":
        result = asyncio.run(test_balance())
    elif choice == "2":
        result = asyncio.run(test_leverage())
    elif choice == "3":
        result = asyncio.run(test_order())
    else:
        print("无效选择")
        sys.exit(1)
    
    print(f"测试结果: {'✅ 成功' if result else '❌ 失败'}")
