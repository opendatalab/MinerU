import os


def get_op_num_threads(env_name: str) -> int:
    env_value = os.getenv(env_name, None)
    return get_op_num_threads_from_value(env_value)


def get_op_num_threads_from_value(env_value: str) -> int:
    if env_value is not None:
        try:
            num_threads = int(env_value)
            if num_threads > 0:
                return num_threads
        except ValueError:
            return -1
    return -1


if __name__ == '__main__':
    print(get_op_num_threads_from_value('1'))
    print(get_op_num_threads_from_value('0'))
    print(get_op_num_threads_from_value('-1'))
    print(get_op_num_threads_from_value('abc'))