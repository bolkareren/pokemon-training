import os
import time

import requests
from bs4 import BeautifulSoup

# Fetch the webpage containing Pokémon sprites
url = "https://pokemondb.net/sprites"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

for img in (
	soup.find_all(string="Generation 1")[0]
	.parent.find_next_sibling()
	.find_all("img", class_="img-fixed icon-pkmn")
):
	alt_name = img.attrs["src"].split("/")[-1].split(".")[0].lower()

	# Create directory for each Pokémon if it doesn't exist
	if not os.path.exists("raw_data/" + alt_name):
		os.makedirs("raw_data/" + alt_name, exist_ok=True)

		# Fetch the individual Pokémon page to get more images
		time.sleep(2)
		url = f"https://pokemondb.net/sprites/{alt_name}"
		response = requests.get(url)
		soup = BeautifulSoup(response.text, "html.parser")

		# Find all images in the "Overview" section
		count = 0
		for img in soup.find("h2", string="Overview").find_next_sibling("div").find_all("img"):
			time.sleep(2)
			response = requests.get(img.attrs["src"])
			if response.status_code == 200:
				try:
					with open(f"raw_data/{alt_name}/image-{count}.png", "wb") as f:
						count += 1
						f.write(response.content)
				except Exception as e:
					print(f"Failed to save image {img.attrs['src']}: {e}")

		print(f"Downloaded {count} images of {alt_name}.")
