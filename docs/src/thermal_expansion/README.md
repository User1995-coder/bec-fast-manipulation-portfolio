# Thermal Expansion Model

## Purpose

`ThermalExpansionModel` converts physical Thomas-Fermi radius expansion
velocities into effective expansion temperatures using the historical analysis
convention of the project.

## Expansion Velocity

The physical input is:

```text
R_dot_i
```

for `i = x, y, z`, expressed in m/s. The model does not know about lambda
scaling factors, Castin-Dum equations, Thomas-Fermi equations, or trap
frequencies.

## Effective Directional Expansion Temperatures

The implemented reference convention is:

```text
T_i = (m / k_B) * R_dot_i^2
```

The result is in K.

## Effective 3D Expansion Temperature

```text
T_3D = (T_x + T_y + T_z) / 3
```

## Historical Convention

An alternative historical convention also appeared in older scripts:

```text
T_i_alt = (m / (3 * k_B)) * R_dot_i^2
T_3D_alt = T_x_alt + T_y_alt + T_z_alt
```

Since `T_i_alt = T_i / 3`, it follows that:

```text
T_3D_alt = T_x/3 + T_y/3 + T_z/3
          = (T_x + T_y + T_z) / 3
          = T_3D
```

The two conventions therefore agree for `T_3D`, but not for directional
temperatures.

## Physical Interpretation

These quantities are effective expansion temperature indicators built from
Thomas-Fermi radius velocities. They reproduce the historical analysis
convention of the project and should not be identified without qualification as
microscopic thermodynamic temperatures.

## Units

```text
velocity    -> m/s
temperature -> K
```

## Usage

```python
from bec_fast_manipulation.thermal_expansion import ThermalExpansionModel

model = ThermalExpansionModel()
temperatures = model.temperatures_from_radius_velocities(vx, vy, vz)
tx = temperatures["x"]
t3d = temperatures["3d"]
```
