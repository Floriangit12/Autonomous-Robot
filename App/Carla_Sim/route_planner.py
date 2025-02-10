# route_planner.py

def generate_route(start_wp, allowed_lane_types, num_steps=50, step_distance=2.0):
    """
    Génère un itinéraire à partir d'un waypoint de départ.
    À chaque étape, on ne conserve que les waypoints dont le type de voie est autorisé.
    """
    route = []
    current_wp = start_wp
    route.append(current_wp)
    for _ in range(num_steps):
        next_wps = current_wp.next(step_distance)
        # Sélectionner uniquement les waypoints sur les voies autorisées
        allowed_next = [wp for wp in next_wps if wp.lane_type in allowed_lane_types]
        if not allowed_next:
            break
        chosen_wp = allowed_next[0]
        route.append(chosen_wp)
        current_wp = chosen_wp
    return route
