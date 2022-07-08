#!/usr/bin/python3

import argparse
import json
import os
from datetime import datetime
from typing import Optional
from urllib.error import HTTPError
from urllib.request import urlopen

# File to save cached events data
cache_path = os.path.join(os.path.expanduser("~"), ".cache", "linkup-cache.json")

event_endpoint = "https://dev-api.linkupevents.com.au/event?id="
events_endpoint = "https://dev-api.linkupevents.com.au/events?uni=unsw"
events_last_updated_endpoint = (
    "https://dev-api.linkupevents.com.au/last-updated?uni=unsw"
)
clubs_paged_endpoint = "https://api.linkupevents.com.au/unsw/clubs?page="

default_events_header = ["name", "hosts", "start", "finish"]
default_clubs_header = ["name", "short_name", "facebook"]

date_format = r"%Y-%m-%d %H:%M"

event_column_value_mappings = {
    "id": lambda event: event["id"],
    "name": lambda event: event["title"],
    "start": lambda event: event["time_start"].strftime(date_format),
    "finish": lambda event: event["time_finish"].strftime(date_format),
    "location": lambda event: event["location"],
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
    "description": lambda event: event["description"].replace("\n", " "),
}

club_column_value_mappings = {
    "short_name": lambda club: (
        club["club_shorthand"] if club["club_shorthand"] != club["club_name"] else ""
    ),
    "name": lambda club: club["club_name"],
    "description": lambda club: club["description"].replace("\n", " "),
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


def valid_columns(columns: list[str], mappings: dict):
    return all(x in mappings for x in columns)


def parse_args():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="mode")

    events_parser = subs.add_parser("events")
    # Search by ID
    events_parser.add_argument("--id", type=int, help="Facebook event ID")
    # or, filter on these fields
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
    events_parser.add_argument("--limit", type=int, help="Maximum rows to display")
    events_parser.add_argument("--sort-by", nargs="+")
    events_parser.add_argument("--columns", nargs="+")

    clubs_parser = subs.add_parser("clubs")
    clubs_parser.add_argument(
        "--page", type=int, default=1, help="Page number (default 1)"
    )
    # Search by ID
    clubs_parser.add_argument("--id", help="Club's ID on LinkUp or Facebook")
    # or, filter on these fields
    clubs_parser.add_argument("--name")
    clubs_parser.add_argument("--categories", nargs="+")
    m = clubs_parser.add_mutually_exclusive_group()
    m.add_argument(
        "--has-fees", action="store_true", help="Show clubs with member fees"
    )
    m.add_argument(
        "--no-fees", action="store_true", help="Show clubs without member fees"
    )
    clubs_parser.add_argument(
        "--search",
        nargs="+",
        help="Match club if any term is in name, categories or description",
    )
    clubs_parser.add_argument("--limit", type=int, help="Maximum rows to display")
    clubs_parser.add_argument("--sort-by", nargs="+")
    clubs_parser.add_argument("--columns", nargs="+")

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
        with open(cache_path, "r") as f:
            cached = json.load(f)
        if cached["last_updated"] != last_updated:
            return None
        return cached["events"]
    except Exception:
        return None


def save_cached_events(last_updated: str, raw_events: list[dict]):
    try:
        with open(cache_path, "w") as f:
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


def object_to_table_row(
    obj: dict, columns: list, mappings: dict, max_width: Optional[int] = None
) -> list[str]:
    if not columns:
        raise ValueError
    row_values = []
    for column in columns:
        mapf = mappings[column]
        displayed_text = mapf(obj)
        if max_width and max_width > 10:
            displayed_text = displayed_text[:max_width]
        row_values.append(displayed_text)
    return row_values


def pprint_table(
    table: list[list[str]], has_header: bool = False, has_footer: bool = False
):
    table = list(table)
    col_width = [max(len(x) for x in col) for col in zip(*table)]
    table_width = sum(col_width) + 3 * len(col_width) + 1

    header = None
    if has_header:
        header = table.pop(0)

    footer = None
    if has_footer:
        footer = table.pop()

    if len(table) == 0:
        table.append(["" for _ in col_width])

    pprint_row = lambda row: print(
        "| "
        + " | ".join("{:{}}".format(x, col_width[i]) for i, x in enumerate(row))
        + " |"
    )

    if header:
        print("-" * table_width)
        pprint_row(header)
        print("-" * table_width)

    for line in table:
        pprint_row(line)

    print("-" * table_width)

    if footer:
        pprint_row(footer)
        print("-" * table_width)


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


def main(args: argparse.Namespace):
    try:
        table = []
        header = []
        footer = None
        page_no = None
        page_count = None
        if args.mode == "events":
            header.extend(args.columns if args.columns else default_events_header)
            table.append(header)
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
                        event, header, event_column_value_mappings, max_width=40
                    )
                    table.append(row)
        elif args.mode == "clubs":
            header.extend(args.columns if args.columns else default_clubs_header)
            table.append(header)
            clubs, page_no, page_count = fetch_clubs_paged(args.page)
            clubs = sort_objects(clubs, args, club_column_value_mappings, "name")
            for club in clubs:
                row = object_to_table_row(club, header, club_column_value_mappings)
                table.append(row)

        if page_no and page_count:
            footer = ["" for _ in header]
            table.append(footer)
            footer[0] = f"Page {args.page} / {page_count}"

        pprint_table(
            table, has_header=header is not None, has_footer=footer is not None
        )
    except RequestError as e:
        print(e.message)
        exit(1)


if __name__ == "__main__":
    main(parse_args())
