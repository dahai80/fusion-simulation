try:
    from fusion_simulation.service.proto import simulation_pb2, simulation_pb2_grpc
except ImportError:
    simulation_pb2 = None  # type: ignore
    simulation_pb2_grpc = None  # type: ignore
