# The LinkUp CLI
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Because why not?!

[Check out the LinkUp website!](https://linkupevents.com/)

> Started at the end of 2019 by UNSW students, Linkup is a dedicated event tracker created in response to the lack of visibility of society events. Our goal is to centralise events so that it's easier to find and navigate society events around campus. To create a place to find new events or societies that you may otherwise not know about, as well as a way to interact with the community through the wide variety of events.

## Requirements

Python 3.7+

## Usage

### Events

Available columns:
- id
- facebook
- name
- start
- finish
- location
- hosts
- categories
- image
- description

```sh
# Get event using its Facebook ID - facebook.com/events/{ID}
linkup events --id 1234567890

# Get all events
linkup events

# Filter events by matching attributes that contain something
linkup events --name bbq
linkup events --host csesoc

# Filter events by searching multiple attributes
# (name, hosts, categories, description)
linkup events --search "free food"

# Filter events by datetime
# Argument should be in ISO format
linkup events --before 2022-07-01T09:00
linkup events --after 2022-07-01T15:30
linkup events --date 2022-07-01

# Selecting specific columns
# Default columns: ["name", "hosts", "start", "facebook"]
linkup events --select start name description facebook

# Sorting by specific columns
# Default sort: "start" ascending
#
# Column names can also be prefixed with underscore _
# to mark it as descending order
linkup events --order-by hosts
linkup events --order-by hosts name
linkup events --order-by _hosts name

# Limiting displayed output
# Default: none
linkup events --limit 10

# Don't like emojis in event descriptions??
linkup events --select name description --no-emojis

# Force the output to use a specific output mode
# Default: It will choose based on the length of printed lines
#
# "table" is best suited for short lines
# "lines" is for long lines
# It is hardcoded with max width of 120 characters
linkup events -o table
linkup events -o lines

# More complete example
linkup events \
    --select name facebook \
    --after 2022-07-01 \
    --search bbq \
    --order-by name
```

### Clubs

Available columns:
- short_name
- name
- description
- tags
- email
- website
- facebook
- facebook_group

```sh
# Get clubs
# As of now, there are 19 pages with 300+ clubs
linkup clubs
linkup clubs -p 1

# Some functions can still be used
linkup clubs \
    -o lines \
    --select name facebook description \
    --order-by _name \
    --limit 5 \
    --no-emojis
```

Note:
- There are limited filtering/search functions for clubs because the "Get Clubs" API is paged.
- Some fields have been omitted (e.g. membership fees), change the source code if you'd like to include them.
