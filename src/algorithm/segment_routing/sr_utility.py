class SRUtility:
    @staticmethod
    def get_segmented_demands(segments: dict, demands: list) -> list:
        """ returns segmented demands as list with with: [(p, q, d), ...] """
        segmented_demands = list()
        for idx, (s, t, d) in enumerate(demands):
            for p, q in segments[idx]:
                segmented_demands.append((p, q, d))
        return segmented_demands
