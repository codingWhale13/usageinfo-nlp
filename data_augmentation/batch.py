def split_into_batches(data: list, n: int):
    return [data[i : i + n] for i in range(0, len(data), n)]


def encode_batch(data):
    def encode(data, index):
        instructions = []
        result = []
        if (isinstance(data, list) or isinstance(data, dict)) and len(data) == 0:
            return {"steps": data, "index": index}, index, []
        elif isinstance(data, list):
            for i, x in enumerate(data):
                steps, index, result_data = encode(x, index)
                instructions.append({"steps": steps, "index": index, "list_index": i})
                result += result_data
            return instructions, index, result
        elif isinstance(data, dict):
            for key, x in data.items():
                steps, index, result_data = encode(x, index)
                instructions.append({"steps": steps, "index": index, "key": key})
                result += result_data
            return instructions, index, result
        else:
            index += 1
            return {"steps": None, "index": index}, index, [data]

    instructions, _, flat_data = encode(data, -1)
    return instructions, flat_data


def decode_batch(flat_data: list, instructions):
    if isinstance(instructions, list):
        if "key" in instructions[0]:
            result = {}
        elif "list_index" in instructions[0]:
            result = []
        for instruction in instructions:
            if "key" in instruction:
                result[instruction["key"]] = decode_batch(
                    flat_data, instruction["steps"]
                )
            elif "list_index" in instruction:
                result.append(decode_batch(flat_data, instruction["steps"]))
        return result
    else:
        if instructions["steps"] is None:
            return flat_data[instructions["index"]]
        else:
            return instructions["steps"]
