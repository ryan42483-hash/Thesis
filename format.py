"""Utilities for reshaping raw NFL data into per-player season records."""

from datetime import datetime
import pandas as pd

# Fields to keep only once per player-season
ONLY_ONCE_FIELDS = {"season", "team", "player_display_name", "position", "position_group"}

# Stats to drop entirely (special teams, fantasy, metadata)
STAT_EXCLUSIONS = {
    "player_name",
    "kickoff_returns",
    "kickoff_return_yards",
    "fg_made",
    "fg_att",
    "fg_missed",
    "fg_blocked",
    "fg_long",
    "fg_pct",
    "fg_made_0_19",
    "fg_made_20_29",
    "fg_made_30_39",
    "fg_made_40_49",
    "fg_made_50_59",
    "fg_made_60_",
    "fg_missed_0_19",
    "fg_missed_20_29",
    "fg_missed_30_39",
    "fg_missed_40_49",
    "fg_missed_50_59",
    "fg_missed_60_",
    "fg_made_list",
    "fg_missed_list",
    "fg_blocked_list",
    "fg_made_distance",
    "fg_missed_distance",
    "fg_blocked_distance",
    "pat_made",
    "pat_att",
    "pat_missed",
    "pat_blocked",
    "pat_pct",
    "gwfg_made",
    "gwfg_att",
    "gwfg_missed",
    "gwfg_blocked",
    "gwfg_distance",
    "punt_return_yards",
    "punt_returns",
    "special_teams_tds",
    "week",
    "player_id",
    "headshot_url",
    "fantasy_points",
    "fantasy_points_ppr",
    "opponent_team",
    "season_type",
}

SPECIAL_TEAMS_POSITIONS = {"P", "K", "LS"}


def collect_player_stats(pbp_player: pd.DataFrame) -> dict:
    """Build {(player_id, season): {stat_name: [values]}} skipping special teams."""

    player_stats = {}

    for _, row in pbp_player.iterrows():
        if row["position"] in SPECIAL_TEAMS_POSITIONS:
            continue

        player_id = row.get("player_id")
        year = row.get("season")
        key = (player_id, year)
        if player_id is None:
            continue

        player_stats.setdefault(key, {})

        for stat_name, value in row.items():
            if stat_name in STAT_EXCLUSIONS:
                continue

            if stat_name not in player_stats[key]:
                player_stats[key][stat_name] = []

            if stat_name in ONLY_ONCE_FIELDS and player_stats[key][stat_name]:
                continue

            if value is None or pd.isna(value):
                continue

            player_stats[key][stat_name].append(value)

    return player_stats


def average_player_stats(player_stats: dict) -> dict:
    """Average numeric lists per stat; keep first value for non-numeric."""

    averaged = {}
    for player_id, stats in player_stats.items():
        averaged[player_id] = {}
        for stat_name, values in stats.items():
            if stat_name == "season":
                averaged[player_id][stat_name] = int(values[0])
                continue
            if not values:
                continue
            if isinstance(values[0], (int, float)):
                averaged[player_id][stat_name] = sum(values) / len(values)
            else:
                averaged[player_id][stat_name] = values[0]
    return averaged


def attach_snap_counts(formatted_stats: dict, snap_counts: pd.DataFrame):
    """Merge snap count data into the stats dict (in place)."""

    def normalize(value):
        if value is None or pd.isna(value):
            return None
        return str(value).strip().lower()

    player_lookup = {}
    for player_id, stats in formatted_stats.items():
        name = normalize(stats.get("player_display_name"))
        position = normalize(stats.get("position"))
        team = normalize(stats.get("team"))
        year = normalize(stats.get("season"))
        if not all([name, position, team, year]):
            continue
        key = (name, position, team, year)
        player_lookup.setdefault(key, []).append(player_id)

    for _, row in snap_counts.iterrows():
        name = normalize(row.get("player"))
        position = normalize(row.get("position"))
        team = normalize(row.get("team"))
        year = normalize(row.get("season"))

        if not all([name, position, team, year]):
            continue

        offense = row.get("offense_snaps")
        offense_pct = row.get("offense_pct")
        defense = row.get("defense_snaps")
        defense_pct = row.get("defense_pct")
        special = row.get("st_snaps")
        special_pct = row.get("st_pct")

        total_snaps = 0
        for snaps in (offense, defense, special):
            total_snaps += snaps

        player_ids = player_lookup.get((name, position, team, year))
        if not player_ids:
            continue

        for pid in player_ids:
            def add(key, value):
                values = formatted_stats[pid].get(key, [])
                values.append(value)
                formatted_stats[pid][key] = values

            add("total_snaps", total_snaps)
            add("offense_snaps", offense if offense is not None else 0)
            add("defense_snaps", defense if defense is not None else 0)
            add("special_snaps", special if special is not None else 0)
            add("offense_pct", offense_pct if offense_pct is not None else 0)
            add("defense_pct", defense_pct if defense_pct is not None else 0)
            add("special_pct", special_pct if special_pct is not None else 0)


def average_snap_count_fields(formatted_stats: dict) -> dict:
    """Average snap counts and weighted percentages."""

    def compute_average(values):
        if not isinstance(values, list) or not values:
            return values
        return sum(values) / len(values)

    def weighted_pct(pct_list, snap_list):
        if not isinstance(pct_list, list) or not isinstance(snap_list, list):
            return pct_list

        numer = 0.0
        denom = 0.0
        scale = None

        for pct, snaps in zip(pct_list, snap_list):
            if pct is None or pd.isna(pct) or snaps is None or pd.isna(snaps):
                continue

            pct_val = float(pct)
            snaps_val = float(snaps)

            frac = pct_val / 100.0 if pct_val > 1 else pct_val
            if frac <= 0:
                continue

            implied_total = snaps_val / frac
            if implied_total <= 0:
                continue

            numer += snaps_val
            denom += implied_total
            if scale is None:
                scale = 100.0 if pct_val > 1 else 1.0

        if denom == 0:
            return 0.0
        if scale is None:
            scale = 1.0
        return (numer / denom) * scale

    averaged = {}
    for pid, stats in formatted_stats.items():
        averaged[pid] = stats.copy()

        for key in ("total_snaps", "offense_snaps", "defense_snaps", "special_snaps"):
            averaged[pid][key] = compute_average(stats.get(key))

        averaged[pid]["offense_pct"] = weighted_pct(stats.get("offense_pct"), stats.get("offense_snaps"))
        averaged[pid]["defense_pct"] = weighted_pct(stats.get("defense_pct"), stats.get("defense_snaps"))
        averaged[pid]["special_pct"] = weighted_pct(stats.get("special_pct"), stats.get("special_snaps"))

    return averaged


def append_player_bio(formatted_stats: dict, players: pd.DataFrame, years) -> dict:
    """Attach bio info (height, weight, age, experience) to each player-season."""

    def safe_val(val):
        return val if val is not None and not pd.isna(val) else None

    for _, row in players.iterrows():
        gsis_id = safe_val(row.get("gsis_id"))
        if gsis_id is None:
            continue

        birth_date = pd.to_datetime(row.get("birth_date"), errors="coerce")
        rookie_season = safe_val(row.get("rookie_season"))
        height = safe_val(row.get("height"))
        weight = safe_val(row.get("weight"))

        for year in years:
            key = (gsis_id, year)
            if key not in formatted_stats:
                continue

            age_in_season = None
            if pd.notna(birth_date):
                try:
                    season_date = datetime(int(year), 9, 1)
                    age_in_season = (season_date - birth_date.to_pydatetime()).days / 365.25
                except Exception:
                    age_in_season = None

            experience = None
            if rookie_season is not None:
                try:
                    experience = (int(year) - int(rookie_season)) + 1
                    if experience <= 0:
                        experience = None
                except Exception:
                    experience = None

            if height is not None:
                formatted_stats[key]["height"] = height
            if weight is not None:
                formatted_stats[key]["weight"] = weight
            if age_in_season is not None:
                formatted_stats[key]["age"] = int(age_in_season)
            if experience is not None:
                formatted_stats[key]["experience"] = experience

    return formatted_stats
