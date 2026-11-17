"""Small, checked reader for TFRecord Example/TensorProto radio measurements.

Only the documented numeric/byte feature types are supported. TensorFlow is
not needed. TFRecord CRC32C checks are verified before protobuf decoding.
"""
import struct

import google_crc32c
import numpy as np


def varint(data, position):
    value, shift = 0, 0
    while position < len(data) and shift < 70:
        byte = data[position]
        position += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            return value, position
        shift += 7
    raise ValueError("Truncated or oversized protobuf varint")


def fields(data):
    position = 0
    while position < len(data):
        key, position = varint(data, position)
        number, wire = key >> 3, key & 7
        if number == 0:
            raise ValueError("Invalid protobuf field number")
        if wire == 0:
            value, position = varint(data, position)
        elif wire in (1, 5):
            length = 8 if wire == 1 else 4
            value = data[position:position+length]
            position += length
            if len(value) != length:
                raise ValueError("Truncated fixed-width protobuf field")
        elif wire == 2:
            length, position = varint(data, position)
            value = data[position:position+length]
            position += length
            if len(value) != length:
                raise ValueError("Truncated length-delimited protobuf field")
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        yield number, wire, value


def tensor(data):
    items = list(fields(data))
    dtype = next(value for number, _, value in items if number == 1)
    shape_message = next(value for number, _, value in items if number == 2)
    shape = tuple(next(value for number, _, value in fields(dim) if number == 1)
                  for number, _, dim in fields(shape_message) if number == 2)
    content = next(value for number, _, value in items if number == 4)
    if dtype not in (1, 2):
        raise ValueError(f"Unsupported tensor dtype {dtype}")
    values = np.frombuffer(content, dtype="<f4" if dtype == 1 else "<f8")
    if values.size != np.prod(shape, dtype=int):
        raise ValueError("Tensor content does not match declared shape")
    return values.reshape(shape)


def example(payload):
    feature_message = next(value for number, _, value in fields(payload) if number == 1)
    result = {}
    for number, _, entry in fields(feature_message):
        if number != 1:
            continue
        parts = list(fields(entry))
        name = next(value.decode("utf-8") for number, _, value in parts if number == 1)
        feature = next(value for number, _, value in parts if number == 2)
        kind, _, content = next(fields(feature))
        elements = [value for number, _, value in fields(content) if number == 1]
        if kind == 1:
            result[name] = elements[0]
        elif kind == 2:
            result[name] = float(np.frombuffer(elements[0], dtype="<f4")[0])
        else:
            raise ValueError(f"Unsupported Example feature type {kind}")
    return result


def masked_crc(data):
    checksum = google_crc32c.value(data)
    return (((checksum >> 15) | (checksum << 17)) + 0xa282ead8) & 0xffffffff


def read_record(data):
    if len(data) < 16:
        raise ValueError("Truncated TFRecord")
    length = struct.unpack("<Q", data[:8])[0]
    if len(data) != length + 16:
        raise ValueError("TFRecord length mismatch")
    if masked_crc(data[:8]) != struct.unpack("<I", data[8:12])[0]:
        raise ValueError("TFRecord header CRC mismatch")
    if masked_crc(data[12:-4]) != struct.unpack("<I", data[-4:])[0]:
        raise ValueError("TFRecord payload CRC mismatch")
    return example(data[12:-4])
