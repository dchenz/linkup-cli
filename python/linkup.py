import argparse
import json
import os
from datetime import datetime
from typing import Optional

import requests

home_dir = os.path.expanduser("~")
cache_path = os.path.join(home_dir, ".cache", "linkup-cache.json")


def parse_args():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="mode")

    events_parser = subs.add_parser("events")
    # Search by ID
    events_parser.add_argument("--id", type=int, help="Facebook event ID")
    # or, filter on these fields
    events_parser.add_argument("--name")
    m = events_parser.add_mutually_exclusive_group()
    m = m.add_mutually_exclusive_group()
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
    events_parser.add_argument("--limit", type=int, help="Maximum rows to display")

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

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")

    if args.mode is None:
        parser.print_help()
        exit(1)
    return args


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
        return None


def fetch_event_by_id(uid: int):
    response = requests.get(f"https://dev-api.linkupevents.com.au/event?id={uid}")
    if not response.ok:
        raise RequestError(response.text)
    data = response.json()
    return normalise_event(data)


def fetch_all_events():
    response = requests.get(
        f"https://dev-api.linkupevents.com.au/last-updated?uni=unsw"
    )
    last_updated = response.text
    raw_events = load_cached_events(last_updated)
    if not raw_events:
        response = requests.get("https://dev-api.linkupevents.com.au/events?uni=unsw")
        if not response.ok:
            raise RequestError("Failed to load events from API")
        raw_events = response.json()
        save_cached_events(last_updated, raw_events)
    return [normalise_event(e) for e in raw_events]


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
    "description": lambda event: event["description"],
}


def event_to_table_row(event: dict, column_order: list, max_width: int = 40):
    order = column_order if column_order else event_column_value_mappings.keys()
    row = []
    for column in order:
        mapf = event_column_value_mappings[column]
        text = mapf(event)
        displayed_text = text
        if max_width > 10:
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
    if not (args.name or args.host or args.before or args.after or args.date):
        return events
    matched = []
    for event in events:
        if args.name and args.name.lower() not in event["title"].lower():
            continue
        hosts_text = event_column_value_mappings["hosts"](event)
        if args.host and args.host.lower() not in hosts_text.lower():
            continue
        if args.before and event["time_start"] >= args.before:
            continue
        if args.after and event["time_start"] <= args.after:
            continue
        if args.date and event["time_start"].date() != args.date.date():
            continue
        matched.append(event)
    return matched


def main(args: argparse.Namespace):
    header = ["name", "hosts", "start", "finish"]
    try:
        if args.mode == "events":

            table = [header]
            if args.id is not None:
                event = fetch_event_by_id(args.id)
                table.append(event_to_table_row(event, header))
            else:
                events = filter_events(fetch_all_events(), args)
                if args.limit:
                    events = events[: args.limit]
                for event in events:
                    table.append(event_to_table_row(event, header))
            pprint_table(table, has_header=True)

    except RequestError as e:
        print(e.message)
        exit(1)


if __name__ == "__main__":
    main(parse_args())
