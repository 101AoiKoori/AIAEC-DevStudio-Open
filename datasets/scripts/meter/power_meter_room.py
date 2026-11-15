"""Reference-based indoor lighting and backdrop for the power meter preview."""

if __name__ == "__main__":
    raise SystemExit("Task helper is import-only; use the primary entry in datasets/scripts/meter/. Production remains paused.")


from array import array
from math import cos, exp, pi, sin, tau

import bpy


def room_environment_pixels():
    """Return the deterministic pixels used by the station ambient map."""
    width, height = 192, 96
    pixels = array('f')
    for y in range(height):
        latitude = pi * ((y + .5) / height - .5)
        up = sin(latitude)
        ceiling = .10 + .30 * max(up, 0) + .09 * exp(-9 * up**2)
        for x in range(width):
            azimuth = tau * (x + .5) / width
            side = .45 * exp(5 * (cos(azimuth - .35 * pi) * cos(latitude) - 1))
            pixels.extend((ceiling + .70 * side, ceiling + .86 * side, ceiling + side, 1))
    return pixels


def room_world(scene):
    """Embed a synthetic equirectangular ambient map with a cool side lobe."""
    width, height = 192, 96
    image = bpy.data.images.new('Synthetic station room environment', width, height, alpha=False,
                                float_buffer=True)
    pixels = room_environment_pixels()
    image.pixels.foreach_set(pixels)
    image.update()
    image.pack()
    image['source'] = 'Procedural indoor ambient map inspired by vehicle_0103.png'

    world = bpy.data.worlds.new('Station room ambient')
    scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    coordinates = nodes.new('ShaderNodeTexCoord')
    rotate = nodes.new('ShaderNodeVectorRotate')
    rotate.rotation_type = 'X_AXIS'
    rotate.inputs['Angle'].default_value = pi / 2
    environment = nodes.new('ShaderNodeTexEnvironment')
    environment.image = image
    environment.projection = 'EQUIRECTANGULAR'
    background = nodes.get('Background')
    background.inputs['Strength'].default_value = .24
    for node, x in zip((coordinates, rotate, environment, background, nodes.get('World Output')),
                       (-760, -550, -340, -70, 140)):
        node.location = (x, 0)
    links.new(coordinates.outputs['Normal'], rotate.inputs['Vector'])
    links.new(rotate.outputs['Vector'], environment.inputs['Vector'])
    links.new(environment.outputs['Color'], background.inputs['Color'])


def add_room_environment(scene, stage, box, material, add_area):
    """Build the Y-up room and return its backdrop collection for side-view isolation."""
    context = bpy.data.collections.new('Room context')
    stage.children.link(context)
    wall = material('Station warm white walls', (.60, .59, .57), .83)
    ceiling = material('Station ceiling', (.48, .49, .49), .8)
    vinyl = material('Station black vinyl floor', (.025, .030, .039), .32, specular=.38)
    barrier = material('Station dark track barrier', (.018, .020, .024), .66)
    paint = material('Station white track markings', (.57, .57, .55), .63)
    frame = material('Station window frame', (.19, .23, .26), .61)
    curtain = material('Station blue grey curtain', (.064, .101, .150), .89)
    glazing = material('Station bright window', (.57, .72, .87), .42, emission=.9, specular=.12)
    panel = material('Station ceiling panel diffuser', (.81, .84, .86), .63, emission=2.0)

    box(stage, 'GEO-room-vinyl-floor', (0, -5.94, 0), (42, .08, 42), vinyl)
    box(context, 'GEO-room-back-wall', (0, -.14, -9.5), (24, 11.52, .20), wall)
    box(context, 'GEO-room-left-wall', (-11.96, -.14, .35), (.20, 11.52, 19.5), wall)
    box(context, 'GEO-room-ceiling', (0, 5.68, .2), (24, .12, 19.6), ceiling)
    box(context, 'GEO-room-track-barrier', (.4, -4.19, -5.94), (21.6, 3.42, .22), barrier)
    box(context, 'GEO-room-utility-column', (-7.0, -2.9, -3.6), (1.6, 6.0, 1.75), wall, .06)

    wx, wy, wz = 6.20, 1.15, -9.33
    box(context, 'GEO-room-window-frame', (wx, wy, wz), (4.80, 3.72, .19), frame, .035)
    for i, dx in enumerate((-1.50, 0, 1.50), 1):
        box(context, f'GEO-room-window-pane-{i}', (wx + dx, wy, wz + .108),
            (1.39, 3.38, .026), glazing)
    box(context, 'GEO-room-window-crossbar', (wx, wy + .70, wz + .13), (4.61, .10, .07), frame)
    box(context, 'GEO-room-window-sill', (wx, wy - 1.92, wz + .11), (4.92, .18, .57), wall, .025)
    for side in (-1, 1):
        for pleat in range(4):
            x = wx + side * (2.03 + .19 * pleat)
            z = wz + .25 + .07 * (pleat % 2)
            box(context, f'GEO-room-curtain-{side}-{pleat}', (x, wy - .06, z),
                (.235, 3.90, .23), curtain, .055)

    for i, (center, size, yaw) in enumerate((
        ((-4.8, -5.892, .1), (.11, .006, 11.0), -.20),
        ((4.85, -5.892, .7), (.13, .006, 11.5), .31),
        ((6.3, -5.892, -2.4), (3.0, .006, .16), .20),
        ((5.5, -5.892, 2.6), (2.5, .006, .14), .20),
    ), 1):
        line = box(context, f'GEO-room-track-line-{i}', center, size, paint)
        line.rotation_euler.y = yaw

    for name, x, z, energy in (('ceiling-key', -3.4, 3.2, 1120),
                                ('ceiling-fill', 3.9, 1.8, 780)):
        box(context, f'GEO-room-{name}-panel', (x, 5.54, z), (3.4, .08, 1.36), panel, .022)
        light_name = f'LGT-room-{name}'
        add_area(stage, light_name, (x, 5.47, z), (0, -1.5, -.3), energy, 3.35)
        light = stage.objects[light_name].data
        light.shape, light.size_y = 'RECTANGLE', 1.32
        light.color = (1.0, .976, .941)

    light_name = 'LGT-room-side-window'
    add_area(stage, light_name, (7.2, .9, 2.2), (0, -1.3, -.1), 430, 4.4)
    light = stage.objects[light_name].data
    light.shape, light.size_y = 'RECTANGLE', 3.4
    light.color = (.80, .89, 1.0)
    room_world(scene)
    return context
