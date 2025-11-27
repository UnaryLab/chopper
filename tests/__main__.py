from mypy import api

stdout, stderr, ret = api.run(["src/", "--strict"])
print(f"mypy returned: {ret}\n{stdout}{stderr}", end="")
