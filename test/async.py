import trio


async def boil_water():
    print("开始烧水")
    await trio.sleep(300)  # 5分钟后水烧开
    print("水烧开了")


async def dry_clothes():
    print("去晾衣服")
    await trio.sleep(5)
    print("晾衣服结束")


async def drink_water():
    print("去喝水")
    await trio.sleep(3)
    print("喝水结束")


async def main():
    async with trio.open_nursery() as nursery:
        nursery.start_soon(boil_water)
        nursery.start_soon(dry_clothes)
        nursery.start_soon(drink_water)


trio.run(main)
