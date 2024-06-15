import io
import os
import shutil
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import neptune
import neptune.handler
import PIL.Image as Image
import torch
import yaml

config = yaml.safe_load(open("config.yaml"))

prev_run = neptune.init_run(
    project="towards-hi/image-classification",
    api_token=config["neptune_api_token"],
    with_id="IM-76",
    mode="read-only",
)

new_run = neptune.init_run(
    project="towards-hi/image-classification",
    api_token=config["neptune_api_token"],
    name="new_run",
    tags=["new_run"],

)
print('this is prev_run', prev_run)


# Helper function to handle different attribute types

f = open('unknown_attributes.txt', 'w')


def copy_attribute(attribute: neptune.handler.Handler, path):
    if hasattr(attribute, 'download'):
        # print('hasattr')
        # Assuming it's a File type
        # file_path = attribute.download()
        # new_run[path].upload(file_path)
        print("not copying file")
    elif hasattr(attribute, 'fetch_values'):
        # Assuming it's a Series type
        print('in elif')
        data_df = attribute.fetch_values()
        print('in elif 2\n', data_df)

        for index, row in data_df.iterrows():
            # print('this is row\n', row["value"], row["step"], type(row["timestamp"]))
            unix_time = row["timestamp"].tz_localize(
                'America/Los_Angeles').tz_convert('UTC').timestamp()
            print('appending to path', path)
            new_run[path].append(
                row["value"], step=row["step"], timestamp=unix_time)
        # print(data['step'], data['value'])
        # for row in data:
            # print('this is row\n', row, data)
            # new_run[path].log(row.value, step=row.step)
    elif hasattr(attribute, 'fetch'):
        print('in else block', path)
        # Assuming it's a single value type (String, Float, Boolean)
        value = attribute.fetch()
        new_run[path] = value
    else:
        print("Unknown attribute type for attribute: ", attribute)
        f.write(f"{path}\n")

        # raise ValueError("Unknown attribute type")

# Recursively copy attributes"


def recurse_copy(attr_dict, path="") -> None:
    for key, item in attr_dict.items():
        new_path = f"{path}/{key}" if path else key
        if isinstance(item, dict):
            print('recursing item', item, new_path)
            recurse_copy(item, path=new_path)
        else:
            print('copying item', item, new_path)
            copy_attribute(item, new_path)
            # return  # for testing


# Fetch the structure of the original run and copy all elements
structure = prev_run.get_structure()
recurse_copy(structure)
