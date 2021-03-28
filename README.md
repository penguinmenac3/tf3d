# tf3d - Transforms for 3D geometry

> Make 3d transformation handling easy.

# Install

It is as simple as a pip install.

```bash
pip install tf3d
```

# Usage

See the full [documentation](docs/README.md).

In a nutshell you import tf3d and use its functions.

```python
import tf3d

P_matrix = tf3d.pinhole_projection(fx, fy, cx, cy)
points = np.ones(22, 3) # 22 points with 3 coordinates x,y,z

uvd = tf3d.project(P_matrix, points)
print(uvd.shape)
xyz = tf3d.reproject(P_matrix, uvd)
print(xyz.shape)

assert (xyz == points).all()
```


# Contribute

You will need to fork the repo, and clone it to your local PC.
Then you must install it in editable mode for development.

```bash
pip install -e .[dev]
```

Now you can fix bugs or add features. When adding features, write the test first and then implement your feature.
Always run the tests.

```bash
python -m nose2
```

Untested code or code that breaks the tests will not be merged.
