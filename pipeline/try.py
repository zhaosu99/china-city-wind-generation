from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GEOJSON_PATH = Path(r"D:\Data\admin\九段线\china.shp")
OUTPUT_PATH = Path(r"D:\HKU PhD\wind data paper\plot\try_china_country_geojson.png")


def main() -> None:
	gdf = gpd.read_file(GEOJSON_PATH)
	if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
		gdf = gdf.to_crs("EPSG:4326")

	fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
	gdf.plot(ax=ax, facecolor="white", edgecolor="black", linewidth=0.8)
	ax.set_title("China Country GeoJSON Preview")
	ax.set_xlabel("Longitude")
	ax.set_ylabel("Latitude")
	ax.set_aspect("equal")
	ax.grid(False)

	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
	plt.close(fig)

	print(f"Saved preview to: {OUTPUT_PATH}")
	print(f"Feature count: {len(gdf)}")
	print(f"Bounds: {tuple(gdf.total_bounds)}")


if __name__ == "__main__":
	main()
