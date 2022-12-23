with open("template.txt", "r") as f:
    config_temp = f.read()
    for fut in [1, 3, 5, 10, 15, 21, 30, 45]:
        config = config_temp.format(fut=fut)
        # save
        with open(f"config_fut_{fut}.yaml", "w") as f:
            f.write(config)
