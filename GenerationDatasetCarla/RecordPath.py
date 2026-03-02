#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import carla
import argparse
import logging
import os
import json
import time
import math
import datetime
from numpy import random
import numpy as np

# ------------------ Helpers Techniques (Path & Ground) ------------------

def get_actor_bottom_z_via_bbox(actor):
    bb = actor.bounding_box
    bottom_local = carla.Location(x=bb.location.x, y=bb.location.y, z=bb.location.z - bb.extent.z)
    return float(actor.get_transform().transform(bottom_local).z)

def raycast_vertical_ground(world, xy_loc, up_offset=0.5, down_distance=10.0):
    best_z = None
    # Petit balayage 3x3 autour de la position pour être précis
    for dx in [-0.1, 0, 0.1]:
        for dy in [-0.1, 0, 0.1]:
            start = carla.Location(x=xy_loc.x + dx, y=xy_loc.y + dy, z=xy_loc.z + up_offset)
            end = carla.Location(x=xy_loc.x + dx, y=xy_loc.y + dy, z=xy_loc.z - down_distance)
            hit = world.cast_ray(start, end)
            if hit and hit[0].location:
                z = float(hit[0].location.z)
                if best_z is None or z > best_z: best_z = z
    return best_z

def get_ground_location_for_ego(world, ego):
    tr = ego.get_transform()
    loc = tr.location
    half_height = float(ego.bounding_box.extent.z)
    
    if "vehicle" in ego.type_id:
        # Pour les voitures : Z d'origine est déjà le sol
        z_sol, z_corr, method = loc.z, 0.0, "car_native_z"
    else:
        # Pour les piétons : Raycast pour trouver le trottoir
        z_val = raycast_vertical_ground(world, loc)
        if z_val is None:
            z_sol, method = loc.z - half_height, "walker_bbox"
        else:
            z_sol, method = z_val, "walker_raycast"
        z_corr = float(z_sol - loc.z)

    return carla.Location(x=loc.x, y=loc.y, z=z_sol), half_height, z_corr, method

def find_best_ego(world, ego_type):
    """ Cherche un acteur qui bouge déjà pour le record """
    filter_str = "walker.pedestrian.*" if ego_type == "walker" else "vehicle.*"
    candidates = [c for c in world.get_actors().filter(filter_str) if c.is_alive]
    if not candidates: return None
    # On privilégie ceux qui ont une vitesse > 0.1 m/s
    moving = [c for c in candidates if c.get_velocity().length() > 0.1]
    return random.choice(moving) if moving else random.choice(candidates)

# ------------------ Main Logic ------------------

def main():
    argparser = argparse.ArgumentParser(description="Spawn NPCs & Record Ego Path")
    argparser.add_argument('--host', default='127.0.0.1')
    argparser.add_argument('--maps', default='Town01', help='Nom de la map CARLA (ex: Town10HD_Opt)')

    argparser.add_argument('-p', '--port', default=2000, type=int)
    argparser.add_argument('-n', '--number-of-vehicles', default=30, type=int)
    argparser.add_argument('-w', '--number-of-walkers', default=50, type=int)
    argparser.add_argument('--ego-type', choices=['walker', 'vehicle'], default='walker')
    argparser.add_argument('--range', default=40.0, type=float, help="Rayon de capture (m)")
    argparser.add_argument('--sample-period', default=0.1, type=float)
    argparser.add_argument('--target-frames', default=1000, type=int)
    argparser.add_argument('--output', default='DataSet/carla_trajectories')
    argparser.add_argument('--sync', action='store_true', help='Mode Synchrone')
    argparser.add_argument('--tm-port', default=8000, type=int)
    argparser.add_argument('--car-lights-on', action='store_true', default=False)
    argparser.add_argument('--nameFile', default='recording.json', help='Nom du fichier de sortie')
    args = argparser.parse_args()
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    walkers_list = []
    all_id = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    synchronous_master = False
    
    try:
        if args.maps:
            available = client.get_available_maps()
            # Accepte un nom court comme "Town10" → cherche la meilleure correspondance
            match = [m for m in available if args.maps in m]
            if not match:
                print(f"❌ Map '{args.maps}' introuvable. Maps disponibles :")
                for m in sorted(available):
                    print(f"   - {m}")
                return
            map_name = match[0]
            print(f"🗺️  Chargement de la map : {map_name}")
            world = client.load_world(map_name)
        else:
            world = client.get_world()
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(5)

        if args.sync:
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)
            synchronous_master = True

        # --- SPAWN VÉHICULES ---
        blueprints = world.get_blueprint_library().filter("vehicle.*")
        blueprints = sorted(blueprints, key=lambda bp: bp.id)
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles: break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                blueprint.set_attribute('color', random.choice(blueprint.get_attribute('color').recommended_values))
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if not response.error: vehicles_list.append(response.actor_id)

        # --- SPAWN PIÉTONS ---
        blueprintsWalkers = world.get_blueprint_library().filter("walker.pedestrian.*")
        spawn_points_w = []
        for i in range(args.number_of_walkers):
            loc = world.get_random_location_from_navigation()
            if loc:
                sp = carla.Transform()
                sp.location = loc
                sp.location.z += 2.0 
                spawn_points_w.append(sp)

        batch = []
        walker_speeds = []
        for sp in spawn_points_w:
            walker_bp = random.choice(blueprintsWalkers)
            if walker_bp.has_attribute('is_invincible'): walker_bp.set_attribute('is_invincible', 'false')
            walker_speeds.append(walker_bp.get_attribute('speed').recommended_values[1] if walker_bp.has_attribute('speed') else 1.4)
            batch.append(SpawnActor(walker_bp, sp))

        results = client.apply_batch_sync(batch, True)
        walker_data = []
        for i in range(len(results)):
            if not results[i].error:
                walker_data.append({"id": results[i].actor_id, "speed": walker_speeds[i]})

        # SPAWN CONTRÔLEURS IA
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for w in walker_data:
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), w["id"]))
        
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if not results[i].error:
                walker_data[i]["con"] = results[i].actor_id
                all_id.extend([results[i].actor_id, walker_data[i]["id"]])

        all_actors = world.get_actors(all_id)
        for _ in range(10): world.tick()

        for i in range(0, len(all_id), 2):
            all_actors[i].start()
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            all_actors[i].set_max_speed(float(walker_data[int(i/2)]["speed"]))

        print(f'✨ Spawn terminé : {len(vehicles_list)} véhicules, {len(walker_data)} piétons.')

        # ------------------ BOUCLE DE RECORDING ------------------
        if not os.path.exists(args.output): os.makedirs(args.output)
        
        ego = find_best_ego(world, args.ego_type)
        out_json = os.path.join(args.output, f"{args.nameFile}.json")
        
        next_sample_t = world.get_snapshot().timestamp.elapsed_seconds + args.sample_period
        traj, samples, stuck_frames = [], 0, 0
        last_wall_time = time.time()
        actors_data = []

        # --- VARIABLES POUR CALCUL MANUEL VITESSE ---
        last_ego_loc = ego.get_location() if ego else None
        last_sim_time = world.get_snapshot().timestamp.elapsed_seconds
        v_manual_x, v_manual_y, v_manual_z = 0.0, 0.0, 0.0
        manual_speed = 0.0

        while samples < args.target_frames:
            if synchronous_master: world.tick()
            else: world.wait_for_tick()

            snap = world.get_snapshot()
            sim_t = snap.timestamp.elapsed_seconds
            dt = sim_t - last_sim_time

            # --- CALCUL DE LA VITESSE MANUELLE ---
            if ego and ego.is_alive and dt > 0:
                curr_loc = ego.get_location()
                if last_ego_loc:
                    v_manual_x = (curr_loc.x - last_ego_loc.x) / dt
                    v_manual_y = (curr_loc.y - last_ego_loc.y) / dt
                    v_manual_z = (curr_loc.z - last_ego_loc.z) / dt
                    manual_speed = math.sqrt(v_manual_x**2 + v_manual_y**2 + v_manual_z**2)
                
                last_ego_loc = curr_loc
                last_sim_time = sim_t
            else:
                manual_speed = 0.0

            # --- SÉCURITÉ RESPAWN ---
            is_stuck = False
            if ego and ego.is_alive:
                # Utilisation de manual_speed au lieu de l'API velocity
                if manual_speed < 0.05: 
                    stuck_frames += 1
                else: 
                    stuck_frames = 0
                if stuck_frames > 3000: is_stuck = True 
            else: 
                is_stuck = True

            if is_stuck:
                print(f"\n♻️ Ego {args.ego_type} bloqué ou disparu. Recherche remplaçant...")
                ego = find_best_ego(world, args.ego_type)
                time.sleep(1.0)
                last_ego_loc = ego.get_location() if ego else None
                stuck_frames = 0
                if not ego: continue

            # FPS Réel
            t_now = time.time()
            fps = 1.0 / (t_now - last_wall_time) if (t_now - last_wall_time) > 0 else 0
            last_wall_time = t_now

            if sim_t >= next_sample_t:
                tr = ego.get_transform()
                gl, gh, zc, gm = get_ground_location_for_ego(world, ego)

                # Filtrage Range
                actors_data = []
                potential = list(world.get_actors().filter("vehicle.*")) + list(world.get_actors().filter("walker.pedestrian.*"))
                for a in potential:
                    if a.id == ego.id: continue
                    a_loc = a.get_location()
                    if tr.location.distance(a_loc) <= args.range:
                        a_tr = a.get_transform()
                        actors_data.append({
                            "actor_id": int(a.id),
                            "type": a.type_id,
                            "dist": round(tr.location.distance(a_loc), 2),
                            "location": {"x": a_loc.x, "y": a_loc.y, "z": a_loc.z},
                            "rotation": {"yaw": a_tr.rotation.yaw, "pitch": a_tr.rotation.pitch, "roll": a_tr.rotation.roll},
                            "velocity": {"x": a.get_velocity().x, "y": a.get_velocity().y, "z": a.get_velocity().z}
                        })

                # FORMAT JSON
                traj.append({
                    "timestamp_sim": float(sim_t),
                    "ego_id": ego.id,
                    "ego_type": ego.type_id,
                    "ego_location": {"x": gl.x, "y": gl.y, "z": gl.z},
                    "ego_body_location": {"x": tr.location.x, "y": tr.location.y, "z": tr.location.z},
                    "ego_rotation": {"yaw": tr.rotation.yaw, "pitch": tr.rotation.pitch, "roll": tr.rotation.roll},
                    "ego_z_correction": zc,
                    "ego_velocity_manual": {"x": v_manual_x, "y": v_manual_y, "z": v_manual_z, "speed": manual_speed},
                    "other_actors": actors_data
                })
                samples += 1
                next_sample_t += args.sample_period

            print(f"\r[{samples}/{args.target_frames}] FPS: {fps:.1f} | Ego: {ego.id} | Stuck: {stuck_frames} | Vit: {manual_speed:.2f} m/s", end="")

        # Save
        with open(out_json, "w") as f: json.dump({"positions": traj}, f, indent=2)
        print(f"\n💾 Dataset sauvé : {out_json}")

    finally:
        if synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)

        print('\n🧹 Cleanup des acteurs...')
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list + all_id])
        time.sleep(0.5)

if __name__ == '__main__':
    main()