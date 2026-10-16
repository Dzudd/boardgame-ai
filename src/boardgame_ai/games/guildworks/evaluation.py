"""Terminal outcome from public scores, independent of incremental rewards."""

from boardgame_ai.core import GameEngine
from boardgame_ai.games.guildworks import GuildworksObservation
from boardgame_ai.simulation.evaluation import GameOutcome


def guildworks_outcome(game: GameEngine[GuildworksObservation]) -> GameOutcome:
    if game.spec.game_id != "guildworks" or game.num_players != 2:
        raise ValueError("Requires a two-seat Guildworks game")
    if not game.is_terminal():
        raise ValueError("Outcome requires a terminal game")
    scores = game.observe(0).data.scores
    tied = scores[0] == scores[1]
    return GameOutcome(
        scores=tuple(map(float, scores)),
        winners=() if tied else (0 if scores[0] > scores[1] else 1,),
        draw=tied,
    )
