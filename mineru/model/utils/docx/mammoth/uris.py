def uri_to_zip_entry_name(base, uri):
    if uri.startswith("/"):
        return uri[1:]
    else:
        return base + "/" + uri


def replace_fragment(uri, fragment):
    hash_index = uri.find("#")
    if hash_index != -1:
        uri = uri[:hash_index]
    return uri + "#" + fragment
