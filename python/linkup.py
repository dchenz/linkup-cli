#!/usr/bin/python3

import argparse
import json
import os
import re
from datetime import datetime
from typing import Optional
from urllib.error import HTTPError
from urllib.request import urlopen

# File to save cached events data
cache_path = os.path.join(os.path.expanduser("~"), ".cache", "linkup")
os.makedirs(cache_path, exist_ok=True)

cache_events_path = os.path.join(cache_path, "events.json")

website_url = "https://linkupevents.com"
event_endpoint = "https://dev-api.linkupevents.com.au/event?id="
events_endpoint = "https://dev-api.linkupevents.com.au/events?uni=unsw"
events_last_updated_endpoint = (
    "https://dev-api.linkupevents.com.au/last-updated?uni=unsw"
)
clubs_paged_endpoint = "https://api.linkupevents.com.au/unsw/clubs?page="

default_events_header = ["name", "hosts", "start", "facebook"]
default_clubs_header = ["name", "short_name", "facebook"]

date_format = r"%Y-%m-%d %H:%M"

# Might be set to True inside parse_args()
no_emoji_output = False

event_column_value_mappings = {
    "id": lambda event: event["id"],
    "facebook": lambda event: "https://www.facebook.com/events/" + event["id"],
    "name": lambda event: event["title"],
    "start": lambda event: event["time_start"].strftime(date_format),
    "finish": lambda event: event["time_finish"].strftime(date_format),
    "location": lambda event: (event["location"] if event["location"] else ""),
    "hosts": lambda event: ", ".join(
        map(
            lambda x: x["name"],
            (
                event["hosts_no_image"]
                if len(event["hosts_image"]) == 0
                else event["hosts_image"]
            ),
        )
    ),
    "categories": lambda event: ", ".join(event["categories"]),
    "image": lambda event: event["image_url"],
    "description": lambda event: (event["description"] if event["description"] else ""),
}

club_column_value_mappings = {
    "short_name": lambda club: (
        club["club_shorthand"] if club["club_shorthand"] != club["club_name"] else ""
    ),
    "name": lambda club: club["club_name"],
    "description": lambda club: (club["description"] if club["description"] else ""),
    "tags": lambda club: ", ".join(club["tags"]),
    "email": lambda club: (
        club["socials"]["email"] if "email" in club["socials"] else ""
    ),
    "website": lambda club: (
        club["socials"]["website"] if "website" in club["socials"] else ""
    ),
    "facebook": lambda club: (
        club["socials"]["facebook_page"] if "facebook_page" in club["socials"] else ""
    ),
    "facebook_group": lambda club: (
        club["socials"]["facebook_group"] if "facebook_group" in club["socials"] else ""
    ),
}

remove_repeated_symbols = re.compile(r"[\*\-\_]{3,}")
remove_repeated_whitespace = re.compile(r"\s +")


def cleanse_text(s: str):
    if no_emoji_output:
        # Replace emojis with spaces
        s = "".join(map(lambda x: x if ord(x) < 256 else " ", s))
    # No newlines
    s = s.replace("\n", " ")
    # Annoying repeated symbols
    s = re.sub(remove_repeated_symbols, " ", s)
    # Repeated whitespace
    return re.sub(remove_repeated_whitespace, " ", s).strip()


def valid_columns(columns: list[str], mappings: dict):
    return all(x in mappings for x in columns)


def parse_args():
    def add_common_args(subparser: argparse.ArgumentParser):
        subparser.add_argument("--limit", type=int, help="Maximum rows to display")
        subparser.add_argument("--sort-by", nargs="+")
        subparser.add_argument("--columns", nargs="+")
        subparser.add_argument(
            "--no-emojis", action="store_true", help="Remove emojis from output"
        )
        subparser.add_argument(
            "-o",
            "--output",
            choices=["table", "lines"],
            help="Default will choose best option",
        )

    help_text = (
        f"Welcome to the LinkUp CLI. A web version is available at {website_url}"
    )
    parser = argparse.ArgumentParser(description=help_text)
    subs = parser.add_subparsers(dest="mode")

    help_text = f"A web version is available at {website_url}/events"
    events_parser = subs.add_parser("events", description=help_text)
    add_common_args(events_parser)

    # Searching by ID ignores other filtering options
    events_parser.add_argument("--id", type=int, help="Facebook event ID")
    events_parser.add_argument("--name")
    m = events_parser.add_mutually_exclusive_group()
    m.add_argument(
        "--before", type=datetime.fromisoformat, help="Starts before 2022-01-31T09:30"
    )
    m.add_argument(
        "--after", type=datetime.fromisoformat, help="Starts after 2022-01-31T09:30"
    )
    m.add_argument(
        "--date",
        type=datetime.fromisoformat,
        help="Starts on 2022-01-31 (time is ignored)",
    )
    events_parser.add_argument("--host")
    events_parser.add_argument(
        "--search",
        nargs="+",
        help="Match event if any term is in name, hosts or description",
    )

    help_text = f"A web version is available at {website_url}/clubs"
    clubs_parser = subs.add_parser("clubs", description=help_text)
    add_common_args(clubs_parser)

    clubs_parser.add_argument(
        "--page", type=int, default=1, help="Page number (default 1)"
    )

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        exit(1)

    mappings = (
        event_column_value_mappings
        if args.mode == "events"
        else club_column_value_mappings
    )

    if getattr(args, "limit", None) is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")

    if getattr(args, "search", None) is not None:
        if getattr(args, "name", None) or getattr(args, "host", None):
            parser.error("--search cannot be used with --name or --host")

    if getattr(args, "sort_by", None) is not None:
        try:
            # Some column names can have underscore prefix
            # for descending sort. Convert args into
            # list[tuple[str, bool]] with sort direction.
            args.sort_by = validate_sort_columns(args.sort_by, mappings)
        except ValueError as e:
            parser.error(str(e))

    if getattr(args, "page", None) is not None and args.page <= 0:
        parser.error("--page must be a positive integer")

    if getattr(args, "columns", None) is not None:
        if valid_columns(args.columns, mappings):
            # Remove duplicate columns
            args.columns = list(dict.fromkeys(args.columns))
        else:
            parser.error("Valid columns: " + ", ".join(mappings.keys()))

    if getattr(args, "no_emojis", None) is not None and args.no_emojis:
        global no_emoji_output
        no_emoji_output = True

    return args


def validate_sort_columns(sort_by: list[str], mappings: dict) -> list[tuple[str, bool]]:
    # Strip prefixes and determine sort direction
    sort_by_directions = []
    check_dupes = {}
    for col in sort_by:
        if col.startswith("_"):
            col = col[1:]
            if check_dupes.get(col) == False:
                raise ValueError(
                    f"Cannot sort both ascending and descending for '{col}'"
                )
            if col not in check_dupes:
                sort_by_directions.append((col, True))
                check_dupes[col] = True
        else:
            if check_dupes.get(col) == True:
                raise ValueError(
                    f"Cannot sort both ascending and descending for '{col}'"
                )
            if col not in check_dupes:
                sort_by_directions.append((col, False))
                check_dupes[col] = False

    if not valid_columns([x[0] for x in sort_by_directions], mappings):
        raise ValueError("Valid columns: " + ", ".join(mappings.keys()))
    return sort_by_directions


class RequestError(Exception):
    def __init__(self, message: str = ""):
        self.message = message


def load_cached_events(last_updated: str) -> Optional[list]:
    try:
        with open(cache_events_path, "r") as f:
            cached = json.load(f)
        if cached["last_updated"] != last_updated:
            return None
        return cached["events"]
    except Exception:
        return None


def save_cached_events(last_updated: str, raw_events: list[dict]):
    try:
        with open(cache_events_path, "w") as f:
            json.dump({"last_updated": last_updated, "events": raw_events}, f)
    except Exception:
        pass


def fetch_event_by_id(uid: int) -> dict:
    try:
        with urlopen(event_endpoint + str(uid)) as response:
            data = json.loads(response.read().decode("utf-8"))
            return normalise_event(data)
    except HTTPError:
        raise RequestError("Event not found")


def fetch_all_events() -> list[dict]:
    last_updated = fetch_last_update_time()
    if last_updated:
        raw_events = load_cached_events(last_updated)
    else:
        raw_events = None
    if not raw_events:
        try:
            with urlopen(events_endpoint) as response:
                raw_events = json.loads(response.read().decode("utf-8"))
                if last_updated:
                    save_cached_events(last_updated, raw_events)
        except HTTPError:
            raise RequestError("Failed to load events from API")
    return [normalise_event(e) for e in raw_events]


def fetch_last_update_time() -> Optional[str]:
    try:
        with urlopen(events_last_updated_endpoint) as response:
            return response.read().decode("utf-8")
    except HTTPError:
        pass


def normalise_event(event: dict) -> dict:
    event["time_start"] = datetime.fromisoformat(event["time_start"])
    event["time_finish"] = datetime.fromisoformat(event["time_finish"])
    event["hosts_image"] = []
    event["hosts_no_image"] = []
    for host in event["hosts"]:
        if host["image"] is None or host["image"].endswith("/default.jpg"):
            event["hosts_no_image"].append(host)
        else:
            event["hosts_image"].append(host)
    del event["hosts"]
    return event


def object_to_table_row(obj: dict, columns: list, mappings: dict) -> list[str]:
    if not columns:
        raise ValueError
    row_values = []
    for column in columns:
        mapf = mappings[column]
        displayed_text = cleanse_text(mapf(obj))
        row_values.append(displayed_text)
    return row_values


def pprint_table(
    table: list[list[str]], header: list[str], max_width: int, no_swap: bool = True
):
    # Don't print empty tables
    if len(table) == 0:
        return

    col_width = [max(len(x) for x in col) for col in zip(header, *table)]
    table_width = sum(col_width) + 3 * len(col_width) + 1

    if not no_swap and table_width > max_width:
        pprint_lines(table, header, max_width)
        return

    pprint_row = lambda row: print(
        "| "
        + " | ".join("{:{}}".format(x, col_width[i]) for i, x in enumerate(row))
        + " |"
    )

    print("-" * table_width)
    pprint_row(header)
    print("-" * table_width)

    for line in table:
        pprint_row(line)

    print("-" * table_width)


def pprint_lines(table: list[list[str]], header: list[str], max_width: int):

    # Don't print empty tables
    if len(table) == 0:
        return

    left_col_width = max(len(x) for x in header)
    right_col_width = max_width - left_col_width - 3
    if left_col_width <= 0 or right_col_width <= 0:
        raise ValueError("Invalid columns widths")

    def print_word_buffer(buf: list[str], header_name: Optional[str]):
        line_value = " ".join(buf)
        if header_name is None:
            padding = " " * (left_col_width + 3)
            print(padding + line_value)
        else:
            print(f"{header_name:>{left_col_width}} = {line_value}")

    for line in table:
        for i in range(len(header)):
            buf = []
            words_length = 0
            first_line = True
            if line[i] == "":
                print_word_buffer([], header[i])
                continue
            words = line[i].split(" ")
            for word in words:
                possible_line_length = words_length + len(word) + max(len(buf) - 1, 0)
                if possible_line_length <= right_col_width:
                    buf.append(word)
                else:
                    # Flush buffer, then add current word
                    print_word_buffer(buf, header[i] if first_line else None)
                    buf.clear()
                    buf.append(word)
                    words_length = 0
                    first_line = False
                words_length += len(word)
            # There are still words in the buffer
            if words_length > 0:
                print_word_buffer(buf, header[i] if first_line else None)
        print()


def filter_events(events: list[dict], args: argparse.Namespace) -> list[dict]:
    if not (
        args.name or args.host or args.before or args.after or args.date or args.search
    ):
        return events
    matched = []
    for event in events:
        if args.name and args.name.lower() not in event["title"].lower():
            continue
        if args.host:
            hosts_text = event_column_value_mappings["hosts"](event)
            if args.host.lower() not in hosts_text.lower():
                continue
        if args.search:
            hosts_text = event_column_value_mappings["hosts"](event)
            name_text = event_column_value_mappings["name"](event)
            desc_text = event_column_value_mappings["description"](event)
            category_text = event_column_value_mappings["categories"](event)
            search_text = " ".join([hosts_text, name_text, desc_text, category_text])
            any_matches = False
            for term in args.search:
                if term.lower() in search_text.lower():
                    any_matches = True
                    break
            if not any_matches:
                continue
        if args.before and event["time_start"] >= args.before:
            continue
        if args.after and event["time_start"] <= args.after:
            continue
        if args.date and event["time_start"].date() != args.date.date():
            continue
        matched.append(event)
    return matched


def sort_objects(
    objs: list[dict], args: argparse.Namespace, mappings: dict, default_key: str
) -> list[dict]:
    # Sort by default
    if not args.sort_by:
        return sorted(objs, key=mappings[default_key])

    def create_sort_key(obj: dict) -> tuple:
        keys = []
        for col, reverse in args.sort_by:
            text_value = mappings[col](obj)
            key = tuple((-ord(x) if reverse else ord(x)) for x in text_value)
            keys.append(key)
        return tuple(keys)

    # Otherwise, sort by custom ordering
    return sorted(objs, key=create_sort_key)


def fetch_clubs_paged(page: int) -> tuple[list[dict], int, int]:
    # Page argument is 1-indexed
    try:
        with urlopen(clubs_paged_endpoint + str(page - 1)) as response:
            response_data = json.loads(response.read())
            if not response_data["is_success"]:
                raise RequestError("Failed to load clubs from API")
            raw_clubs = response_data["clubs"]
            page_count = response_data["nbPages"]
            if page > page_count:
                return fetch_clubs_paged(page_count)
            return [normalise_club(x) for x in raw_clubs], page, page_count
    except HTTPError:
        raise RequestError("Failed to load clubs from API")


def normalise_club(club: dict) -> dict:
    club["tags"].extend(club["categories"])
    club["tags"].sort()
    del club["categories"]
    return club


def main(args: argparse.Namespace, max_width: int):
    try:
        table = []
        header = []
        page_no = None
        page_count = None
        if args.mode == "events":
            header.extend(args.columns if args.columns else default_events_header)
            if args.id is not None:
                event = fetch_event_by_id(args.id)
                row = object_to_table_row(event, header, event_column_value_mappings)
                table.append(row)
            else:
                events = filter_events(fetch_all_events(), args)
                events = sort_objects(
                    events, args, event_column_value_mappings, "start"
                )
                if args.limit:
                    events = events[: args.limit]
                for event in events:
                    row = object_to_table_row(
                        event, header, event_column_value_mappings
                    )
                    table.append(row)
        elif args.mode == "clubs":
            header.extend(args.columns if args.columns else default_clubs_header)
            clubs, page_no, page_count = fetch_clubs_paged(args.page)
            clubs = sort_objects(clubs, args, club_column_value_mappings, "name")
            for club in clubs:
                row = object_to_table_row(club, header, club_column_value_mappings)
                table.append(row)

        if args.output == "lines":
            pprint_lines(table, header, max_width)
        else:
            pprint_table(table, header, max_width, no_swap=args.output == "table")

        print(f"Total results: {len(table)}")

        if len(table) > 0 and page_no and page_count:
            print(f"Page: {page_no} of {page_count}")

    except RequestError as e:
        print(e.message)
        exit(1)


if __name__ == "__main__":
    main(parse_args(), 120)
