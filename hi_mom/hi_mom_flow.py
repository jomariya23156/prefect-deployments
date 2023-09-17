import numpy as np
from prefect import flow, task

@task
def say_hello():
    print("Hi Mom!")
    print("Test required package:",np.array([1,2,3]))

@flow(name="Hi Mom Flow")
def hi_mom_flow():
    say_hello()