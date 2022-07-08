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


def valid_columns(columns: list[str]):
    return all(x in event_column_value_mappings for x in columns)


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
    events_parser.add_argument("--columns", nargs="+", help="Columns to display")

    clubs_parser = subs.add_parser("clubs")
    # Search by ID
    clubs_parser.add_argument("--id", help="Club's ID on LinkUp or Facebook")
    # or, filter on these fields
    clubs_parser.add_argument("--name")
    clubs_parser.add_argument("--description")
    clubs_parser.add_argument("--category")
    clubs_parser.add_argument("--tags", nargs="+")
    clubs_parser.add_argument("--fees-arc", type=int)
    clubs_parser.add_argument("--fees-non-arc", type=int)
    clubs_parser.add_argument("--fees-associate", type=int)

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        exit(1)

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")

    if args.search is not None and (args.name or args.host):
        parser.error("--search cannot be used with --name or --host")

    if args.sort_by is not None:
        try:
            # Some column names can have underscore prefix
            # for descending sort. Convert args into
            # list[tuple[str, bool]] with sort direction.
            args.sort_by = validate_sort_columns(args.sort_by)
        except ValueError as e:
            parser.error(str(e))

    if args.columns is not None:
        if valid_columns(args.columns):
            args.columns = list(set(args.columns))
        else:
            parser.error(
                "Valid columns: " + ", ".join(event_column_value_mappings.keys())
            )

    return args


def validate_sort_columns(sort_by_columns: list[str]) -> list[tuple[str, bool]]:
    # Strip prefixes and determine sort direction
    sort_by_directions = []
    check_dupes = {}
    for col in sort_by_columns:
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

    if not valid_columns([x[0] for x in sort_by_directions]):
        raise ValueError(
            "Valid columns: " + ", ".join(event_column_value_mappings.keys())
        )
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


def normalise_event(event: dict):
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


def event_to_table_row(
    event: dict, column_order: list, max_width: Optional[int] = None
):
    order = column_order if column_order else event_column_value_mappings.keys()
    row = []
    for column in order:
        mapf = event_column_value_mappings[column]
        text = mapf(event)
        displayed_text = text
        if max_width and max_width > 10:
            displayed_text = displayed_text[:max_width]
        row.append(displayed_text)
    return row


def pprint_table(table: list[list[str]], has_header: bool = False):
    col_width = [max(len(x) for x in col) for col in zip(*table)]
    for line in table:
        line_content = (
            "| "
            + " | ".join("{:{}}".format(x, col_width[i]) for i, x in enumerate(line))
            + " |"
        )
        if has_header:
            print("-" * len(line_content))
        print(line_content)
        if has_header:
            print("-" * len(line_content))
            has_header = False


def filter_events(events: list[dict], args: argparse.Namespace):
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


def sort_events(events: list[dict], args: argparse.Namespace):
    # Sort by time_start by default
    if not args.sort_by:
        return sorted(events, key=event_column_value_mappings["start"])

    def create_sort_key(event: dict) -> tuple:
        keys = []
        for col, reverse in args.sort_by:
            text_value = event_column_value_mappings[col](event)
            key = tuple((-ord(x) if reverse else ord(x)) for x in text_value)
            keys.append(key)
        return tuple(keys)

    # Otherwise, sort by custom ordering
    return sorted(events, key=create_sort_key)


def main(args: argparse.Namespace):
    header = args.columns if args.columns else ["name", "hosts", "start", "finish"]
    try:
        if args.mode == "events":

            table = [header]
            if args.id is not None:
                event = fetch_event_by_id(args.id)
                table.append(event_to_table_row(event, header))
            else:
                events = filter_events(fetch_all_events(), args)
                events = sort_events(events, args)
                if args.limit:
                    events = events[: args.limit]
                for event in events:
                    table.append(event_to_table_row(event, header, max_width=40))
            pprint_table(table, has_header=True)

    except RequestError as e:
        print(e.message)
        exit(1)


if __name__ == "__main__":
    main(parse_args())
