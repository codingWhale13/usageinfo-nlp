def wait_for_debugger(port: int = 56783):
    """
    Pauses the program until a remote debugger is attached. Should only be called on rank0.
    """

    import debugpy

    debugpy.listen(("0.0.0.0", port))
    print(
        f"Waiting for client to attach on port {port}... NOTE: if using docker, you need to forward the port with -p {port}:{port}."
    )
    debugpy.wait_for_client()