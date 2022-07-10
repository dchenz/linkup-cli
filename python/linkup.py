#!/usr/bin/python3

import argparse
import json
import os
import re
from datetime import datetime
from typing import Optional, Union
from urllib.error import HTTPError
from urllib.request import Request, urlopen

# Public endpoints for LinkUp API
website_url = "https://linkupevents.com"
event_endpoint = "https://dev-api.linkupevents.com.au/event?id="
events_endpoint = "https://dev-api.linkupevents.com.au/events?uni=unsw"
events_last_updated_endpoint = (
    "https://dev-api.linkupevents.com.au/last-updated?uni=unsw"
)
clubs_paged_endpoint = "https://api.linkupevents.com.au/unsw/clubs?page="

# File to save cached events data
cache_path = os.path.join(os.path.expanduser("~"), ".cache", "linkup")
cache_events_path = os.path.join(cache_path, "events.json")

# These columns must be @property in their respective classes
default_events_header = ["name", "hosts", "start", "facebook"]
default_clubs_header = ["name", "short_name", "facebook"]
default_event_sort = "start"
default_club_sort = "name"

date_format = r"%Y-%m-%d %H:%M"


class RequestError(Exception):
    def __init__(self, message: str = ""):
        self.message = message


class Event:

    _property_names = None

    @staticmethod
    def _columns() -> set[str]:
        # Get @property method names only
        # Other methods should have _ prefix
        if Event._property_names is None:
            props = (x for x in dir(Event) if not x.startswith("_"))
            Event._property_names = set(props)
        return Event._property_names

    @staticmethod
    def _validate_columns(columns: list[str]) -> bool:
        cs = Event._columns()
        return all(x in cs for x in columns)

    def __init__(self, event_data: dict):
        event_data = dict(event_data)
        event_data["time_start"] = datetime.fromisoformat(event_data["time_start"])
        event_data["time_finish"] = datetime.fromisoformat(event_data["time_finish"])
        event_data["hosts_image"] = []
        event_data["hosts_no_image"] = []
        for host in event_data["hosts"]:
            if host["image"] is None or host["image"].endswith("/default.jpg"):
                event_data["hosts_no_image"].append(host)
            else:
                event_data["hosts_image"].append(host)
        del event_data["hosts"]
        self.event = event_data

    @property
    def id(self) -> str:
        return self.event["id"]

    @property
    def facebook(self) -> str:
        return "https://www.facebook.com/events/" + self.event["id"]

    @property
    def name(self) -> str:
        return self.event["title"]

    @property
    def start(self) -> str:
        return self.event["time_start"].strftime(date_format)

    @property
    def finish(self) -> str:
        return self.event["time_finish"].strftime(date_format)

    @property
    def location(self) -> str:
        return self.event["location"] if self.event["location"] else ""

    @property
    def hosts(self) -> str:
        hosts = self.event["hosts_image"]
        if len(hosts) == 0:
            hosts = self.event["hosts_no_image"]
        host_names = map(lambda x: x["name"], hosts)
        return ", ".join(host_names)

    @property
    def categories(self) -> str:
        return ", ".join(self.event["categories"])

    @property
    def image(self) -> str:
        return self.event["image_url"]

    @property
    def description(self) -> str:
        return self.event["description"] if self.event["description"] else ""


class Club:

    _property_names = None

    @staticmethod
    def _columns() -> set[str]:
        # Get @property method names only
        # Other methods should have _ prefix
        if Club._property_names is None:
            props = (x for x in dir(Club) if not x.startswith("_"))
            Club._property_names = set(props)
        return Club._property_names

    @staticmethod
    def _validate_columns(columns: list[str]) -> bool:
        cs = Club._columns()
        return all(x in cs for x in columns)

    def __init__(self, club_data: dict):
        club_data = dict(club_data)
        club_data["tags"].extend(club_data["categories"])
        club_data["tags"].sort()
        del club_data["categories"]
        self.club = club_data

    @property
    def short_name(self) -> str:
        if self.club["club_shorthand"] == self.club["club_name"]:
            return ""
        return self.club["club_shorthand"]

    @property
    def name(self) -> str:
        return self.club["club_name"]

    @property
    def description(self) -> str:
        return self.club["description"]

    @property
    def tags(self) -> str:
        return ", ".join(self.club["tags"])

    @property
    def website(self) -> str:
        return self.club["socials"].get("website", "")

    @property
    def email(self) -> str:
        return self.club["socials"].get("email", "")

    @property
    def facebook(self) -> str:
        return self.club["socials"].get("facebook_page", "")

    @property
    def facebook_group(self) -> str:
        return self.club["socials"].get("facebook_group", "")


ClubOrEvent = Union[Club, Event]


remove_repeated_symbols = re.compile(r"[\*\-\_]{3,}")
remove_repeated_whitespace = re.compile(r"\s +")


def cleanse_text(s: str, no_emojis: bool) -> str:
    if no_emojis:
        # Replace emojis with spaces
        s = "".join(map(lambda x: x if ord(x) < 256 else " ", s))
    # No newlines
    s = s.replace("\n", " ")
    # Annoying repeated symbols
    s = re.sub(remove_repeated_symbols, " ", s)
    # Repeated whitespace
    return re.sub(remove_repeated_whitespace, " ", s).strip()


def parse_args() -> argparse.Namespace:
    def add_common_args(subparser: argparse.ArgumentParser):
        subparser.add_argument("--limit", type=int, help="Maximum rows to display")
        subparser.add_argument("--order-by", nargs="+")
        subparser.add_argument("--columns", "--select", nargs="+")
        subparser.add_argument(
            "--no-emojis", action="store_true", help="Remove emojis from output"
        )
        subparser.add_argument(
            "-o",
            "--output",
            choices=["table", "lines"],
            help="Default will choose best option",
        )

    # ---- Main parser ----

    help_text = (
        f"Welcome to the LinkUp CLI. A web version is available at {website_url}"
    )
    parser = argparse.ArgumentParser(description=help_text)
    subs = parser.add_subparsers(dest="mode")

    # ---- Events subparser ----

    help_text = f"A web version is available at {website_url}/events"
    events_parser = subs.add_parser("events", description=help_text)
    add_common_args(events_parser)

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

    # ---- Clubs subparser ----

    help_text = f"A web version is available at {website_url}/clubs"
    clubs_parser = subs.add_parser("clubs", description=help_text)
    add_common_args(clubs_parser)

    clubs_parser.add_argument(
        "-p", "--page", type=int, default=1, help="Page number (default 1)"
    )

    # ---- End parsing and validate ----

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        exit(1)

    if args.mode == "events":
        validate_columns = Event._validate_columns
        available_columns = sorted(Event._columns())
    else:
        validate_columns = Club._validate_columns
        available_columns = sorted(Club._columns())

    if getattr(args, "limit", None) is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")

    if getattr(args, "search", None) is not None:
        if getattr(args, "name", None) or getattr(args, "host", None):
            # --search already includes text from "name" and "hosts"
            # hence it doesn't make sense to have them together
            parser.error("--search cannot be used with --name or --host")

    if getattr(args, "order_by", None) is not None:
        try:
            # Some column names can have underscore prefix
            # for descending sort. Convert args into
            # list[tuple[str, bool]] with sort direction.
            args.order_by = validate_sort_columns(args.order_by)
            if not validate_columns([x[0] for x in args.order_by]):
                raise ValueError("Valid columns: " + ", ".join(available_columns))
        except ValueError as e:
            parser.error(str(e))

    if getattr(args, "page", None) is not None and args.page <= 0:
        parser.error("--page must be a positive integer")

    if getattr(args, "columns", None) is not None:
        if validate_columns(args.columns):
            # Remove duplicate columns
            args.columns = list(dict.fromkeys(args.columns))
        else:
            parser.error("Valid columns: " + ", ".join(available_columns))

    return args


def validate_sort_columns(order_by: list[str]) -> list[tuple]:
    # Strip prefixes and determine sort direction
    order_by_directions = []
    check_dupes = {}
    for col in order_by:
        if col.startswith("_"):
            col = col[1:]
            if check_dupes.get(col) == False:
                raise ValueError(
                    f"Cannot sort both ascending and descending for '{col}'"
                )
            if col not in check_dupes:
                order_by_directions.append((col, True))
                check_dupes[col] = True
        else:
            if check_dupes.get(col) == True:
                raise ValueError(
                    f"Cannot sort both ascending and descending for '{col}'"
                )
            if col not in check_dupes:
                order_by_directions.append((col, False))
                check_dupes[col] = False

    return order_by_directions


def load_cached_events(last_updated: str) -> Optional[list]:
    try:
        with open(cache_events_path, "r") as f:
            cached = json.load(f)
        if cached["last_updated"] != last_updated:
            return None
        return cached["events"]
    except Exception:
        # If cache fails, fetch new data from API
        # Should be transparent to user
        pass


def save_cached_events(last_updated: str, raw_events: list[dict]):
    try:
        with open(cache_events_path, "w") as f:
            json.dump({"last_updated": last_updated, "events": raw_events}, f)
    except Exception:
        # Should be transparent to user
        pass


def fetch_event_by_id(uid: int) -> Event:
    try:
        response_text = request_get(event_endpoint + str(uid))
        raw_event = json.loads(response_text)
        return Event(raw_event)
    except HTTPError:
        raise RequestError("Event not found")


def fetch_all_events() -> tuple[list[Event], Optional[str]]:
    last_updated = fetch_last_update_time()

    if last_updated:
        raw_events = load_cached_events(last_updated)
    else:
        raw_events = None

    if not raw_events:
        try:
            response_text = request_get(events_endpoint)
            raw_events = json.loads(response_text)
            if last_updated:
                save_cached_events(last_updated, raw_events)
        except HTTPError:
            raise RequestError("Failed to load events from API")

    return [Event(e) for e in raw_events], last_updated


def fetch_last_update_time() -> Optional[str]:
    try:
        s = request_get(events_last_updated_endpoint)
        return s.strip('"')
    except HTTPError:
        pass


def request_get(url: str) -> str:
    r = Request(url)
    r.add_header("User-Agent", "linkup-cli")
    with urlopen(r) as response:
        return response.read().decode("utf-8")


def to_table_row(obj: ClubOrEvent, columns: list[str], no_emojis: bool) -> list[str]:
    # Must have at least one column
    if not columns:
        raise ValueError

    row_values = []
    for column in columns:
        cell_value = getattr(obj, column)
        displayed_text = cleanse_text(cell_value, no_emojis)
        row_values.append(displayed_text)

    return row_values


def pprint_table(
    table: list[list[str]], header: list[str], max_width: int, no_swap: bool = True
):
    # Don't print empty tables
    if len(table) == 0:
        return

    # List of column maximum widths to determine extra spaces to print
    col_width = [max(len(x) for x in col) for col in zip(header, *table)]
    table_width = sum(col_width) + 3 * len(col_width) + 1

    # Swap to "lines" mode if table exceeds max_width
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


def filter_events(events: list[Event], args: argparse.Namespace) -> list[Event]:
    # No filtering arguments used so just return here
    if not (
        args.name or args.host or args.before or args.after or args.date or args.search
    ):
        return events

    matched = []
    for event in events:
        if args.name and args.name.lower() not in event.name.lower():
            continue
        if args.host:
            if args.host.lower() not in event.hosts.lower():
                continue
        if args.search:
            search_text = " ".join(
                [event.hosts, event.name, event.description, event.categories]
            )
            any_matches = False
            for term in args.search:
                if term.lower() in search_text.lower():
                    any_matches = True
                    break
            if not any_matches:
                continue
        if args.before and event.event["time_start"] >= args.before:
            continue
        if args.after and event.event["time_start"] <= args.after:
            continue
        if args.date and event.event["time_start"].date() != args.date.date():
            continue
        matched.append(event)

    return matched


def sort_objects(
    objs: list[ClubOrEvent], args: argparse.Namespace, default_key: str
) -> list[ClubOrEvent]:

    # Sort by default
    if not args.order_by:
        return sorted(objs, key=lambda x: getattr(x, default_key))

    # Sort strings as tuples of their ord(x) values
    def create_sort_key(obj: ClubOrEvent) -> tuple:
        keys = []
        for col, reverse in args.order_by:
            text_value = getattr(obj, col)
            key = tuple((-ord(x) if reverse else ord(x)) for x in text_value)
            keys.append(key)
        return tuple(keys)

    # Sort by custom ordering
    return sorted(objs, key=create_sort_key)


def fetch_clubs_paged(page: int) -> tuple[list[Club], int, int]:
    # Page argument is 1-indexed
    try:
        response_text = request_get(clubs_paged_endpoint + str(page - 1))
        response_data = json.loads(response_text)
        if not response_data["is_success"]:
            raise RequestError("Failed to load clubs from API")
        raw_clubs = response_data["clubs"]
        page_count = response_data["nbPages"]
        if page > page_count:
            return fetch_clubs_paged(page_count)
        return [Club(x) for x in raw_clubs], page, page_count
    except HTTPError:
        raise RequestError("Failed to load clubs from API")


def main(args: argparse.Namespace, max_width: int):
    try:
        table = []
        header = []
        page_no = None
        page_count = None
        last_update = None
        if args.mode == "events":
            header.extend(args.columns if args.columns else default_events_header)
            if args.id is not None:
                event = fetch_event_by_id(args.id)
                row = to_table_row(event, header, args.no_emojis)
                table.append(row)
            else:
                events, last_update = fetch_all_events()
                events = filter_events(events, args)
                events = sort_objects(events, args, default_event_sort)  # type: ignore
                if args.limit:
                    events = events[: args.limit]
                for event in events:
                    row = to_table_row(event, header, args.no_emojis)
                    table.append(row)
        elif args.mode == "clubs":
            header.extend(args.columns if args.columns else default_clubs_header)
            clubs, page_no, page_count = fetch_clubs_paged(args.page)
            clubs = sort_objects(clubs, args, default_club_sort)  # type: ignore
            for club in clubs:
                row = to_table_row(club, header, args.no_emojis)
                table.append(row)

        if args.output == "lines":
            pprint_lines(table, header, max_width)
        else:
            pprint_table(table, header, max_width, no_swap=args.output == "table")

        print(f"Total results: {len(table)}")

        if len(table) > 0 and page_no and page_count:
            print(f"Page: {page_no} of {page_count}")

        if last_update:
            print(f"Last updated: {last_update}")

    except RequestError as e:
        print(e.message)
        exit(1)


if __name__ == "__main__":
    os.makedirs(cache_path, exist_ok=True)
    main(parse_args(), 120)
