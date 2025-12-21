# Adding telemetry

Adding a module here allows for collecting new types of telemetry.

Telemetry modules are expected to have a "main" function with the following arguments:

`main(stop: bool, *args, **kwargs)`

where stop is controlled by the telemetry runner.
