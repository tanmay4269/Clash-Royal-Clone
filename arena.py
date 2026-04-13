from utils import *
from collections import deque

from player_side import PlayerSide, PlayerSide1, PlayerSide2
from entity import Entity

from entities.building import Building
from entities.troop import Troop

from entities.troops.knight import Knight
from entities.troops.giant import Giant
from entities.troops.mini_pekka import MiniPEKKA


class Arena:
    def __init__(self):
        self.tile_size = 16  # Sub-tile cells
            # * Each cell is just one pixel as off now, temporary simplificaton

        self.width = 18  # In tiles
        self.height = 32

        self.objects: Set[Entity] = set()  # Contains all in game objects that have been deployed like buildings, troupes, etc.
        self.deploy_buffer = set()         # Contains items that haven't deployed yet but need to be rendered

        # Occupancy Grid
        #   0 => Unoccupied
        #   1 => Permanent occupancy
        #   2 => Building
        #   3 => Troop
        # TODO: This should be its own class coz I later also wanna implement an `on_update`
        #   that depends on this matrix being updated and only upon update down stream recomputation occurs
        self.cell_occupancy = np.zeros(
            (self.width * self.tile_size, 
            self.height * self.tile_size), 
        )

        # Top and bottom rows
        self.occupy_cells(np.ones((self.tile_size * 6, self.tile_size)), (0, 0))
        self.occupy_cells(np.ones((self.tile_size * 6, self.tile_size)), (self.tile_size * 12, 0))
        self.occupy_cells(np.ones((self.tile_size * 6, self.tile_size)), (0, (self.height-1) * self.tile_size))
        self.occupy_cells(np.ones((self.tile_size * 6, self.tile_size)), (self.tile_size * 12, (self.height-1) * self.tile_size))

        # The divider
        self.occupy_cells(np.ones((self.tile_size * 5//2, self.tile_size * 2)), (0, self.tile_size * 15))
        self.occupy_cells(np.ones((self.tile_size * 5//2, self.tile_size * 2)), (self.tile_size * 31//2, self.tile_size * 15))
        self.occupy_cells(np.ones((self.tile_size * 9, self.tile_size * 2)),    (self.tile_size * 9//2, self.tile_size * 15))
        
        # Adding player sides
        self.player_side_1 = PlayerSide1()  # The one closer to (0, 0)
        self.player_side_2 = PlayerSide2()

        self.player_side_1.set_opponent(self.player_side_2)
        self.player_side_2.set_opponent(self.player_side_1)

        # Deploying crown towers
        towers = self.player_side_1.get_objects() | self.player_side_2.get_objects()
        for obj in towers:
            self.deploy_entity(obj)


        self.elapsed_time = 0
        self.game_duration = 90  # sec
        self._font = None

        # * DEBUG *
        self._debug_active_player = 1  # For spawning the troop on the right side of the arena
        self._debug_active_card = Knight

    
    def render(self, screen, render_cell_occupancy=True) -> None:
        """        
        * For simplicity, I'll keep each sub-tile cell as one pixel
        """

        # Ground layer
        screen.fill("#D1CC95")

        # Occupancy map
        if render_cell_occupancy:
            # TODO: Make it so that this isn't wastefuly recomputed each time
            rgb_occupancy_map = np.empty((self.cell_occupancy.shape[0], self.cell_occupancy.shape[1], 3), dtype=np.uint8)

            # TODO: Manage multiple layers
            rgb_occupancy_map[:, :, 0] = 255 * (1 - self.cell_occupancy)
            rgb_occupancy_map[:, :, 1] = 255 * (1 - self.cell_occupancy)
            rgb_occupancy_map[:, :, 2] = 255 * (1 - self.cell_occupancy)

            surface = pygame.surfarray.make_surface(rgb_occupancy_map)
            surface.set_alpha(127)
            screen.blit(surface, (0, 0))


        # Faint gridlines 
        #   TODO: This can probably be optimised by precomputing a sprite 
        for r in range(self.height):
            for c in range(self.width):
                pygame.draw.line(screen, (128, 128, 128), (0, self.tile_size * r), (self.tile_size * self.width, self.tile_size * r), 1)
                pygame.draw.line(screen, (128, 128, 128), (self.tile_size * c, 0), (self.tile_size * c, self.tile_size * self.height), 1)


        # All objects
        for obj in self.objects:
            obj.render(screen)

        for obj in self.deploy_buffer:
            obj.render(screen)

        
        # Highlighted cell under the cursor
        (mouse_x, mouse_y) = pygame.mouse.get_pos()
        tile_x = mouse_x // self.tile_size
        tile_y = mouse_y // self.tile_size

        surface = pygame.Surface((self.tile_size, self.tile_size))
        surface.set_alpha(127)
        surface.fill((128, 128, 128))
        screen.blit(surface, (tile_x * self.tile_size, tile_y * self.tile_size))

        # HUD: elixir (bottom left) and timer (bottom right)
        if self._font is None:
            self._font = pygame.font.SysFont(None, 14)

        screen_w = self.width * self.tile_size
        screen_h = self.height * self.tile_size

        elixir_text_1 = self._font.render(f"E: {self.player_side_1.elixirs:.0f}", True, (220, 220, 220))
        screen.blit(elixir_text_1, (4, 4))

        elixir_text_2 = self._font.render(f"E: {self.player_side_2.elixirs:.0f}", True, (220, 220, 220))
        screen.blit(elixir_text_2, (4, screen_h - elixir_text_2.get_height() - 4))

        timer_text = self._font.render(f"{int(self.elapsed_time)}s", True, (220, 220, 220))
        screen.blit(timer_text, (screen_w - timer_text.get_width() - 4, 4))
 

    def update(self, dt) -> Tuple[bool, bool]:
        """
        return: 
            terminated: if someone won
            truncated: if time limit exceeded and the game hasnt terminated
        """

        self.elapsed_time += dt
        if self.elapsed_time >= self.game_duration:
            return False, True

        ### Collision Management ###

        # Makes a reasonable simplifying assumption that buildings are rects and troops are circles
        # TODO: Maybe much later I implement spacial proximity based approach. If things lag, this could be an optimisation
        # TODO: Put this in another method
        for obj_i in self.objects:
            for obj_j in self.objects:
                if obj_i == obj_j:
                    continue

                if not isinstance(obj_j, Troop):
                    continue

                if isinstance(obj_i, Building):
                    # Building-troop collision
                    delta = obj_i.position - obj_j.position
                    overlap = (obj_i.size / 2 + Vector2(obj_j.size, obj_j.size)) - Vector2(abs(delta.x), abs(delta.y))

                    if overlap.x < 0 or overlap.y < 0:
                        continue

                    force = -delta.normalize() * overlap.length() * Troop.COLLISION_COEF
                    obj_j.apply_force(force)
                else: 
                    # Troop-troop collision
                    delta = obj_i.position - obj_j.position
                    overlap = (obj_i.size + obj_j.size) - delta.length()

                    if overlap < 0:
                        continue

                    force = delta.normalize() * overlap * Troop.COLLISION_COEF
                    obj_i.apply_force(force)
                    obj_j.apply_force(-force)


        ### Deploy Buffer Management ###
        
        deployed_objs = set()
        for obj in self.deploy_buffer:
            if obj.has_deployed(dt):
                deployed_objs.add(obj)
        
        for obj in deployed_objs:
            self.objects.add(obj)
            self.deploy_buffer.remove(obj)


        ### Object Update and Deletion ###

        dead_objs = set()
        for obj in self.objects:
            if not obj.update(dt, self.cell_occupancy):
                dead_objs.add(obj)

        while len(dead_objs):
            obj = dead_objs.pop()

            if obj.owner.side_index == 1:
                self.player_side_1.remove_object(obj)
            elif obj.owner.side_index == 2:
                self.player_side_2.remove_object(obj)
            
            if obj == self.player_side_1.king_tower:
                print("Player 2 won!")
                return True, False
            elif obj == self.player_side_2.king_tower:
                print("Player 1 won!")
                return True, False

            self.objects.remove(obj)
            del obj


        ### Elixir Update ###
        self.player_side_1.update(dt)
        self.player_side_2.update(dt)
        
        return False, False


    def on_click(self) -> None:
        (mouse_x, mouse_y) = pygame.mouse.get_pos()
        tile_row = mouse_y // self.tile_size
        tile_col = mouse_x // self.tile_size

        # * DEBUG * 
        if self._debug_active_player == 1:
            troop = self._debug_active_card(self.player_side_1, tile_row + 1, tile_col + 1)
            self.player_side_1.add_object(troop)
        else:
            troop = self._debug_active_card(self.player_side_2, tile_row + 1, tile_col + 1)
            self.player_side_2.add_object(troop)
        
        if self.deploy_entity(troop) is False:
            # print("Failed deploying troop")
            ...


    def deploy_entity(self, deploy_me: Entity) -> bool:
        """
        Return false if the entity can't be deployed in its current form
        """

        # 1. Check if player has enough elixir
        if deploy_me.owner.elixirs < deploy_me.deploy_cost:
            return False
        
        # 2. Check if the deploy location (already written into the object, 
        #   access via public method) is available to deploy, if not return False
        mask, mask_pos = deploy_me.get_cell_occupancy()
        if self.occupy_cells(mask, mask_pos) is False:
            return False

        # 3. Add to self.objects
        # self.objects.add(deploy_me)
        self.deploy_buffer.add(deploy_me)

        # 4. Subtract player's elixirs and return True
        deploy_me.owner.spend_elixirs(deploy_me.deploy_cost)

        return True


    def occupy_cells(self, mask: np.ndarray, mask_pos) -> bool:
        """
        If mask overlaps with something, return false
        mask_pos is expected to be x, y and mask is to be width, height shaped
        
        Can also be used to "unoccupy" cells
        """

        # assert len(mask.shape) == 2

        if isinstance(mask_pos, Vector2):
            mask_pos = (int(mask_pos.x), int(mask_pos.y))

        # 1. Check with self.cell_occupancy, if any intersection, return false
        row_min, row_max = mask_pos[0], mask_pos[0] + mask.shape[0]
        col_min, col_max = mask_pos[1], mask_pos[1] + mask.shape[1]

        tmp_mask = self.cell_occupancy[
            row_min : row_max, 
            col_min : col_max, 
        ]

        # Check on each layer
        for bg_layer in range(1, Entity.CELL_OCCUPANCY_LAYERS+1):
            for fg_layer in range(bg_layer, Entity.CELL_OCCUPANCY_LAYERS+1):
                if bg_layer == 3 and bg_layer == fg_layer:
                    continue   # Don't check for troop-troop deployment constraint
                tmp_mask_layer  = np.where(tmp_mask == bg_layer, True, False)
                mask_layer      = np.where(mask == fg_layer, True, False)

                if np.any(tmp_mask_layer & mask_layer):
                    return False

        # 2. Else just OR it to the cell occupancy
        self.cell_occupancy[
            row_min : row_max, 
            col_min : col_max, 
        ] = mask

        return True

