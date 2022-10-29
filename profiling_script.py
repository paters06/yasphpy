def profiling_script(func):
    import cProfile
    import pstats
    profiler = cProfile.Profile()
    profiler.enable()
    func()
    profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('ncalls')
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats()