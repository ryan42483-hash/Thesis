import pandas as pd
from datetime import datetime


only_once = {'season', 'team', 'player_display_name', 'position', 'position_group'}
stat_exclusions = {'player_name', 'kickoff_returns', 'kickoff_return_yards', 'fg_made', 'fg_att', 'fg_missed', 'fg_blocked', 'fg_long', 'fg_pct', 'fg_made_0_19', 'fg_made_20_29', 'fg_made_30_39', 'fg_made_40_49', 'fg_made_50_59', 'fg_made_60_', 'fg_missed_0_19', 'fg_missed_20_29', 'fg_missed_30_39', 'fg_missed_40_49', 'fg_missed_50_59', 'fg_missed_60_', 'fg_made_list', 'fg_missed_list', 'fg_blocked_list', 'fg_made_distance', 'fg_missed_distance', 'fg_blocked_distance', 'pat_made', 'pat_att', 'pat_missed', 'pat_blocked', 'pat_pct', 'gwfg_made', 'gwfg_att', 'gwfg_missed', 'gwfg_blocked', 'gwfg_distance', 'punt_return_yards', 'punt_returns', 'special_teams_tds', 'week','player_id', 'headshot_url', 'fantasy_points', 'fantasy_points_ppr', 'opponent_team', 'season_type'}
special_teams = {'P', 'K', 'LS'}

def format(pbp_player):
    """
    Build a mapping of player_id -> {stat_field: [values]} from the pbp_player dataframe.

    Parameters
    ----------
    pbp_injury : pandas.DataFrame
        Currently unused; kept in the signature for future integration.
    pbp_player : pandas.DataFrame
        Player stats loaded from nflreadpy.

    Returns
    -------
    dict
        {player_id: {stat_name: [values]}}
    """
    player_stats = {}

    for _, row in pbp_player.iterrows():
        if row['position'] in special_teams:
            continue
        player_id = row.get("player_id")
        year = row.get("season")
        identify = (player_id, year)
        if player_id is None:
            continue

        if identify not in player_stats:
            player_stats[identify] = {}

        for stat_name, value in row.items():
            if stat_name in stat_exclusions:
                continue

            if stat_name not in player_stats[identify]:
                player_stats[identify][stat_name] = []

            if stat_name in only_once and player_stats[identify][stat_name]:
                continue  # Only keep the first value for these stats
            else:
                if value is None or pd.isna(value):
                    continue
                else:
                    player_stats[identify][stat_name].append(value)

    return player_stats

def average_stats(player_stats):
    """
    Compute average stats for each player.

    Parameters
    ----------
    player_stats : dict
        {player_id: {stat_name: [values]}}
    """
    averaged_stats = {}

    for player_id, stats in player_stats.items():
        averaged_stats[player_id] = {}
        for stat_name, values in stats.items():
            if stat_name == 'season':
                averaged_stats[player_id][stat_name] = int(values[0])
                continue
            if not values:
                continue
            if isinstance(values[0], (int, float)):
                averaged_stats[player_id][stat_name] = sum(values) / len(values)
            else:
                averaged_stats[player_id][stat_name] = values[0]  # Keep the first value for non-numeric stats

    return averaged_stats

def add_snap_count(formated_stats, snap_counts):
    # nflreadpy uses Polars; convert to pandas if the rest of your code expects pandas
    

    def normalize(value):
        if value is None or pd.isna(value):
            return None
        return str(value).strip().lower()

    # Index formatted stats by (player_display_name, position, team)
    player_lookup = {}
    for player_id, stats in formated_stats.items():
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

        # Treat missing values as zero
        total_snaps = 0
        for snaps in (offense, defense, special):
            total_snaps += snaps

        player_ids = player_lookup.get((name, position, team, year))
        if not player_ids:
            continue
        
        # Accumulate snaps across rows (e.g., per-week entries)
        for pid in player_ids:
            def add(key, value):
                lst = formated_stats[pid].get(key, [])
                lst.append(value)
                formated_stats[pid][key] = lst

            add("total_snaps", total_snaps)
            add("offense_snaps", offense if offense is not None else 0)
            add("defense_snaps", defense if defense is not None else 0)
            add("special_snaps", special if special is not None else 0)
            add("offense_pct", offense_pct if offense_pct is not None else 0)
            add("defense_pct", defense_pct if defense_pct is not None else 0)
            add("special_pct", special_pct if special_pct is not None else 0)

def average_snap_counts(formated_stats):
    def compute_average(values):
        if not isinstance(values, list) or not values:
            return values
        return sum(values) / len(values)

    def weighted_pct(pct_list, snap_list):
        if not isinstance(pct_list, list) or not isinstance(snap_list, list):
            return pct_list

        numer = 0.0
        denom = 0.0
        scale = None  # 1.0 for fraction, 100.0 for percentage

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
    for pid, stats in formated_stats.items():
        averaged[pid] = stats.copy()

        # Average snap counts
        for key in ("total_snaps", "offense_snaps", "defense_snaps", "special_snaps"):
            averaged[pid][key] = compute_average(stats.get(key))

        # Weighted averages for percentages using implied total snaps
        averaged[pid]["offense_pct"] = weighted_pct(
            stats.get("offense_pct"), stats.get("offense_snaps")
        )
        averaged[pid]["defense_pct"] = weighted_pct(
            stats.get("defense_pct"), stats.get("defense_snaps")
        )
        averaged[pid]["special_pct"] = weighted_pct(
            stats.get("special_pct"), stats.get("special_snaps")
        )

    return averaged


def add_personal_info (formated_stats, players, years):
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
            if key not in formated_stats:
                continue

            # Age during season (approximate as of Sept 1st)
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
                formated_stats[key]["height"] = height
            if weight is not None:
                formated_stats[key]["weight"] = weight
            if age_in_season is not None:
                formated_stats[key]["age"] = int(age_in_season)
            if experience is not None:
                formated_stats[key]["experience"] = experience

    return formated_stats
