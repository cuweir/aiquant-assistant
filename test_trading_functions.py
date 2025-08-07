#!/usr/bin/env python3
"""
快速测试交易功能的脚本
测试：下单、止损、平仓的完整流程
"""

import asyncio
import sys
import os
from decimal import Decimal

# Add the app directory to the Python path
sys.path.append(os.path.abspath('.'))

from app.services.order_executor import OrderExecutor
from app.core.config import settings

async def test_trading_functions():
    """测试完整的交易流程"""
    print("=" * 60)
    print("🚀 开始测试交易功能")
    print("=" * 60)
    
    # 使用测试网络
    executor = OrderExecutor(is_testnet=True)
    test_symbol = "BNB/USDT"
    test_leverage = 5
    test_margin_usdt = 21.0  # 使用21 USDT保证金
    
    try:
        # 步骤1: 检查余额
        print("\n📊 步骤1: 检查账户余额")
        balance = await executor.get_balance('USDT')
        print(f"  > 可用余额: {balance:.2f} USDT")
        
        if balance < test_margin_usdt:
            print(f"  ❌ 余额不足，需要至少 {test_margin_usdt} USDT")
            return False
        
        # 步骤2: 设置杠杆
        print(f"\n⚙️ 步骤2: 设置杠杆为 {test_leverage}x")
        leverage_success = await executor.set_leverage(test_symbol, test_leverage)
        if not leverage_success:
            print("  ❌ 设置杠杆失败")
            return False
        print(f"  ✅ 杠杆设置成功: {test_leverage}x")
        
        # 步骤3: 开仓 (使用保证金数量)
        print(f"\n📈 步骤3: 开多头仓位")
        print(f"  > 使用保证金: {test_margin_usdt} USDT")
        print(f"  > 名义价值: {test_margin_usdt * test_leverage} USDT")
        
        entry_order = await executor.create_market_order_by_notional(
            test_symbol, 'buy', test_margin_usdt * test_leverage
        )
        
        if not entry_order:
            print("  ❌ 开仓失败")
            return False
        
        print(f"  ✅ 开仓成功!")
        print(f"    订单ID: {entry_order.get('id')}")
        print(f"    成交数量: {entry_order.get('filled')} {test_symbol.split('/')[0]}")
        print(f"    成交价格: {entry_order.get('price', 'N/A')}")
        print(f"    成交金额: {entry_order.get('cost', 'N/A')} USDT")
        
        # 获取成交信息
        filled_amount = float(entry_order.get('filled', 0))
        entry_price = float(entry_order.get('price', 0))
        
        if filled_amount <= 0:
            print("  ❌ 获取成交数量失败")
            return False
        
        # 步骤4: 设置止损单
        print(f"\n🛡️ 步骤4: 设置止损单")
        stop_loss_price = entry_price * 0.98  # 2%止损
        print(f"  > 止损价格: {stop_loss_price:.4f}")
        
        sl_order = await executor.create_stop_loss_order(
            test_symbol, 'sell', filled_amount, stop_loss_price
        )
        
        if not sl_order:
            print("  ❌ 止损单设置失败")
            # 如果止损失败，立即平仓保护资金
            print("  🚨 止损失败，立即平仓保护资金...")
            await executor.close_market_position(test_symbol, 'LONG', filled_amount)
            return False
        
        print(f"  ✅ 止损单设置成功!")
        print(f"    止损订单ID: {sl_order.get('id')}")
        
        # 步骤5: 等待几秒钟模拟持仓
        print(f"\n⏳ 步骤5: 模拟持仓 (等待5秒)")
        await asyncio.sleep(5)
        
        # 步骤6: 取消止损单
        print(f"\n❌ 步骤6: 取消止损单")
        cancel_success = await executor.cancel_order(sl_order['id'], test_symbol)
        if cancel_success:
            print(f"  ✅ 止损单取消成功")
        else:
            print(f"  ⚠️ 止损单取消失败 (可能已经执行)")
        
        # 步骤7: 手动平仓
        print(f"\n📉 步骤7: 手动平仓")
        close_order = await executor.close_market_position(test_symbol, 'LONG', filled_amount)
        
        if not close_order:
            print("  ❌ 平仓失败")
            return False
        
        print(f"  ✅ 平仓成功!")
        print(f"    平仓订单ID: {close_order.get('id')}")
        print(f"    平仓数量: {close_order.get('filled')} {test_symbol.split('/')[0]}")
        
        # 步骤8: 检查最终余额
        print(f"\n💰 步骤8: 检查最终余额")
        final_balance = await executor.get_balance('USDT')
        balance_change = final_balance - balance
        print(f"  > 最终余额: {final_balance:.2f} USDT")
        print(f"  > 余额变化: {balance_change:+.2f} USDT")
        
        print("\n" + "=" * 60)
        print("🎉 所有交易功能测试完成!")
        print("✅ 测试结果: 成功")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        print("🔧 建议检查:")
        print("  1. 网络连接是否正常")
        print("  2. API密钥是否正确")
        print("  3. 测试网络余额是否充足")
        return False
        
    finally:
        # 关闭连接
        await executor.close_connections()
        print("\n🔌 连接已关闭")

async def quick_balance_check():
    """快速检查余额"""
    print("💰 快速余额检查...")
    executor = OrderExecutor(is_testnet=True)
    try:
        balance = await executor.get_balance('USDT')
        print(f"  > 测试网络余额: {balance:.2f} USDT")
        if balance < 50:
            print("  ⚠️ 余额较低，建议前往 https://testnet.binancefuture.com 获取测试币")
        else:
            print("  ✅ 余额充足，可以进行测试")
    except Exception as e:
        print(f"  ❌ 余额检查失败: {e}")
    finally:
        await executor.close_connections()

if __name__ == "__main__":
    print("请选择测试模式:")
    print("1. 完整交易功能测试 (推荐)")
    print("2. 快速余额检查")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        print("\n⚠️ 注意: 这将在测试网络上执行真实的交易操作")
        confirm = input("确认继续? (y/N): ").strip().lower()
        if confirm == 'y':
            success = asyncio.run(test_trading_functions())
            sys.exit(0 if success else 1)
        else:
            print("测试已取消")
    elif choice == "2":
        asyncio.run(quick_balance_check())
    else:
        print("无效选择")
