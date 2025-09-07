import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="\033[90m%(asctime)s \033[36m[%(levelname)s] \033[1;33m%(module)s\033[0m: %(message)s",
)
