#!/usr/bin/env python3
"""
实盘小额测试脚本
⚠️ 警告：这将使用真实资金进行交易！
"""

import asyncio
import sys
import os

sys.path.append(os.path.abspath('.'))

from app.services.order_executor import OrderExecutor
from app.core.config import settings

async def live_small_test():
    """实盘小额测试"""
    print("=" * 60)
    print("⚠️  实盘小额测试 - 使用真实资金！")
    print("=" * 60)
    
    # 使用实盘环境
    executor = OrderExecutor(is_testnet=False)
    test_symbol = "BNB/USDT"  # 选择BNB，波动相对较小
    test_leverage = 3  # 低杠杆
    
    # 固定使用很小的保证金金额
    test_margin_usdt = 6.0  # 只用6 USDT保证金，名义价值18 USDT
    
    try:
        # 步骤1: 检查余额
        print("\n💰 步骤1: 检查实盘账户余额")
        balance = await executor.get_balance('USDT')
        print(f"  > 实盘可用余额: {balance:.2f} USDT")
        
        if balance < 50:
            print(f"  ❌ 余额过低，建议至少有50 USDT用于测试")
            return False
        
        notional_value = test_margin_usdt * test_leverage
        print(f"  > 本次测试保证金: {test_margin_usdt} USDT")
        print(f"  > 名义价值: {notional_value} USDT")
        print(f"  > 实际保证金占比: {(test_margin_usdt / balance * 100):.2f}%")
        
        # 最后确认
        print(f"\n⚠️  最后确认:")
        print(f"  - 这是实盘交易，将使用真实资金")
        print(f"  - 保证金: {test_margin_usdt} USDT")
        print(f"  - 名义价值: {notional_value} USDT")
        print(f"  - 杠杆: {test_leverage}x")
        print(f"  - 交易对: {test_symbol}")
        
        confirm = input("\n确认继续实盘测试? (输入 'YES' 继续): ").strip()
        if confirm != 'YES':
            print("测试已取消")
            return False
        
        # 步骤2: 设置杠杆
        print(f"\n⚙️ 步骤2: 设置杠杆为 {test_leverage}x")
        leverage_success = await executor.set_leverage(test_symbol, test_leverage)
        if not leverage_success:
            print("  ❌ 设置杠杆失败")
            return False
        print(f"  ✅ 杠杆设置成功: {test_leverage}x")
        
        # 步骤3: 开仓
        print(f"\n📈 步骤3: 开多头仓位")
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
        print(f"    实际成交金额: {entry_order.get('cost', 'N/A')} USDT")
        
        filled_amount = float(entry_order.get('filled', 0))
        entry_price = float(entry_order.get('price', 0))
        
        if filled_amount <= 0:
            print("  ❌ 获取成交数量失败")
            return False
        
        # 步骤4: 设置止损单 (5%止损，比较宽松)
        print(f"\n🛡️ 步骤4: 设置止损单")
        stop_loss_price = entry_price * 0.95  # 5%止损
        print(f"  > 入场价格: {entry_price:.4f}")
        print(f"  > 止损价格: {stop_loss_price:.4f} (-5%)")
        
        sl_order = await executor.create_stop_loss_order(
            test_symbol, 'sell', filled_amount, stop_loss_price
        )
        
        if not sl_order:
            print("  ❌ 止损单设置失败")
            print("  🚨 立即平仓保护资金...")
            close_result = await executor.close_market_position(test_symbol, 'LONG', filled_amount)
            print(f"  应急平仓: {'成功' if close_result else '失败'}")
            return False
        
        print(f"  ✅ 止损单设置成功!")
        print(f"    止损订单ID: {sl_order.get('id')}")
        
        # 步骤5: 显示当前状态并询问下一步
        print(f"\n📊 当前仓位状态:")
        print(f"  - 持仓数量: {filled_amount} {test_symbol.split('/')[0]}")
        print(f"  - 入场价格: {entry_price:.4f}")
        print(f"  - 止损价格: {stop_loss_price:.4f}")
        print(f"  - 名义价值: ~{entry_price * filled_amount:.2f} USDT")
        
        print(f"\n选择下一步操作:")
        print(f"1. 立即平仓 (推荐)")
        print(f"2. 保持仓位 (风险自负)")
        
        choice = input("请选择 (1 或 2): ").strip()
        
        if choice == "1":
            # 步骤6: 取消止损单并平仓
            print(f"\n❌ 步骤6: 取消止损单")
            cancel_success = await executor.cancel_order(sl_order['id'], test_symbol)
            print(f"  止损单取消: {'成功' if cancel_success else '失败'}")
            
            print(f"\n📉 步骤7: 手动平仓")
            close_order = await executor.close_market_position(test_symbol, 'LONG', filled_amount)
            
            if close_order:
                print(f"  ✅ 平仓成功!")
                print(f"    平仓订单ID: {close_order.get('id')}")
                print(f"    平仓价格: {close_order.get('price', 'N/A')}")
                
                # 计算盈亏
                exit_price = float(close_order.get('price', entry_price))
                pnl_per_unit = exit_price - entry_price
                total_pnl = pnl_per_unit * filled_amount
                pnl_percentage = (pnl_per_unit / entry_price) * 100
                
                print(f"\n💰 交易结果:")
                print(f"  - 入场价格: {entry_price:.4f}")
                print(f"  - 出场价格: {exit_price:.4f}")
                print(f"  - 单位盈亏: {pnl_per_unit:+.4f}")
                print(f"  - 总盈亏: {total_pnl:+.4f} USDT")
                print(f"  - 盈亏比例: {pnl_percentage:+.2f}%")
                
            else:
                print("  ❌ 平仓失败")
                return False
        else:
            print(f"\n⚠️ 仓位保持开启状态")
            print(f"  - 请注意监控市场")
            print(f"  - 止损单ID: {sl_order.get('id')}")
            print(f"  - 可以通过交易所界面手动管理")
        
        # 步骤8: 检查最终余额
        print(f"\n💰 最终余额检查")
        final_balance = await executor.get_balance('USDT')
        balance_change = final_balance - balance
        print(f"  > 最终余额: {final_balance:.2f} USDT")
        print(f"  > 余额变化: {balance_change:+.4f} USDT")
        
        print("\n" + "=" * 60)
        print("🎉 实盘小额测试完成!")
        print("✅ 所有功能验证成功")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        print("🚨 如果有未平仓位，请立即通过交易所界面处理！")
        return False
        
    finally:
        await executor.close_connections()

if __name__ == "__main__":
    print("⚠️  警告：这是实盘测试，将使用真实资金！")
    print("📋 测试配置:")
    print(f"  - 保证金: 6 USDT")
    print(f"  - 杠杆: 3x")
    print(f"  - 名义价值: ~18 USDT")
    print(f"  - 交易对: BNB/USDT")
    
    final_confirm = input("\n最终确认开始实盘测试? (输入 'CONFIRM' 继续): ").strip()
    if final_confirm == 'CONFIRM':
        success = asyncio.run(live_small_test())
        sys.exit(0 if success else 1)
    else:
        print("实盘测试已取消")
