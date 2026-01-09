import matplotlib.pyplot as plt
from typing import List, Union
import numpy as np
from collections import defaultdict


def compute_accuracies_ex2(
    commands: List[Union[str, List[str]]],
    gold_actions: List[Union[str, List[str]]],
    pred_actions: List[Union[str, List[str]]],
):
    assert len(commands) == len(gold_actions) == len(pred_actions), (
        "commands, gold_actions, pred_actions must have the same length"
    )

    # Normalize sequences to lists of tokens
    def to_tokens(x):
        if isinstance(x, str):
            return x.split()
        return x

    commands_tok = [to_tokens(c) for c in commands]
    gold_actions_tok = [to_tokens(g) for g in gold_actions]
    pred_actions_tok = [to_tokens(p) for p in pred_actions]

    n = len(commands_tok)

    # Per-example lengths
    per_example_action_lengths = []
    per_example_command_lengths = []

    # Global token-level counts
    total_correct_tokens = 0
    total_tokens = 0

    # Global sequence-level counts (exact match)
    total_correct_seqs = 0

    # Grouped stats (token-level): [correct_tokens, total_tokens]
    by_action_len = defaultdict(lambda: [0, 0])
    by_cmd_len = defaultdict(lambda: [0, 0])

    # Grouped stats (sequence-level): [correct_seqs, total_seqs]
    by_action_len_seq = defaultdict(lambda: [0, 0])
    by_cmd_len_seq = defaultdict(lambda: [0, 0])

    for cmd_tokens, gold, pred in zip(commands_tok, gold_actions_tok, pred_actions_tok):
        action_len = len(gold)
        cmd_len = len(cmd_tokens)

        per_example_action_lengths.append(action_len)
        per_example_command_lengths.append(cmd_len)

        # ----- sequence-level (exact match) -----
        is_exact_match = pred == gold
        if is_exact_match:
            total_correct_seqs += 1

        # update grouped sequence-level stats
        by_action_len_seq[action_len][1] += 1  # total seqs
        by_cmd_len_seq[cmd_len][1] += 1  # total seqs
        if is_exact_match:
            by_action_len_seq[action_len][0] += 1  # correct seqs
            by_cmd_len_seq[cmd_len][0] += 1  # correct seqs

        # ----- token-level comparison with length penalty -----
        L = max(len(gold), len(pred))
        correct_tokens_example = 0
        total_tokens_example = 0

        for i in range(L):
            total_tokens += 1
            total_tokens_example += 1

            if i < len(gold) and i < len(pred):
                if pred[i] == gold[i]:
                    total_correct_tokens += 1
                    correct_tokens_example += 1
            else:
                # one side missing → automatically wrong
                continue

        by_action_len[action_len][0] += correct_tokens_example
        by_action_len[action_len][1] += total_tokens_example

        by_cmd_len[cmd_len][0] += correct_tokens_example
        by_cmd_len[cmd_len][1] += total_tokens_example

    # Overall token-level accuracy
    token_level_accuracy = (
        (total_correct_tokens / total_tokens) if total_tokens > 0 else 0.0
    )

    # Overall sequence-level accuracy (exact match)
    seq_level_accuracy = (total_correct_seqs / n) if n > 0 else 0.0

    # Accuracy by action length (token-level)
    action_lengths = sorted(by_action_len.keys())
    accuracy_by_action_length = [
        (by_action_len[L][0] / by_action_len[L][1]) if by_action_len[L][1] > 0 else 0.0
        for L in action_lengths
    ]

    # Accuracy by command length (token-level)
    command_lengths = sorted(by_cmd_len.keys())
    accuracy_by_command_length = [
        (by_cmd_len[N][0] / by_cmd_len[N][1]) if by_cmd_len[N][1] > 0 else 0.0
        for N in command_lengths
    ]

    # Sequence-level (exact match) accuracy by action length
    seq_accuracy_by_action_length = [
        (by_action_len_seq[L][0] / by_action_len_seq[L][1])
        if by_action_len_seq[L][1] > 0
        else 0.0
        for L in action_lengths
    ]

    # Sequence-level (exact match) accuracy by command length
    seq_accuracy_by_command_length = [
        (by_cmd_len_seq[N][0] / by_cmd_len_seq[N][1])
        if by_cmd_len_seq[N][1] > 0
        else 0.0
        for N in command_lengths
    ]

    return {
        "token_level_accuracy": token_level_accuracy,
        "seq_level_accuracy": seq_level_accuracy,
        "per_example_action_lengths": per_example_action_lengths,
        "per_example_command_lengths": per_example_command_lengths,
        "action_lengths": action_lengths,
        "token_accuracy_by_action_length": [a * 100 for a in accuracy_by_action_length],
        "seq_accuracy_by_action_length": [
            a * 100 for a in seq_accuracy_by_action_length
        ],
        "command_lengths": command_lengths,
        "token_accuracy_by_command_length": [
            c * 100 for c in accuracy_by_command_length
        ],
        "seq_accuracy_by_command_length": [
            c * 100 for c in seq_accuracy_by_command_length
        ],
    }


def plot_ex2_tokenAccSeqLength(x_val, y_val):
    plt.figure(figsize=(10, 6))

    plt.bar(x_val, y_val, width=0.8)

    plt.title("Token-Level Accuracy by Action Sequence Length", fontsize=16)
    plt.xlabel("Ground-Truth Action Sequence Length (in words)", fontsize=14)
    plt.ylabel("Accuracy on New Commands (%)", fontsize=14)

    plt.grid(axis="y", linestyle="-", color="lightgray")

    plt.ylim(0, 100)
    plt.tight_layout()

    plt.show()
    plt.savefig("ex2_token_acc_by_seq_length.png")


def plot_ex2_seqAccSeqLength(x_val, y_val):
    plt.figure(figsize=(10, 6))

    plt.bar(x_val, y_val, width=0.8)

    plt.title("Sequence-Level Accuracy by Action Sequence Length", fontsize=16)
    plt.xlabel("Ground-Truth Action Sequence Length (in words)", fontsize=14)
    plt.ylabel("Accuracy on New Commands (%)", fontsize=14)

    plt.grid(axis="y", linestyle="-", color="lightgray")

    plt.ylim(0, 100)
    plt.tight_layout()

    plt.show()
    plt.savefig("ex2_seq_acc_by_seq_length.png")


def plot_ex2_tokenAccCommandLength(x_val, y_val):
    plt.figure(figsize=(10, 6))
    positions = np.arange(len(x_val))

    plt.bar(positions, y_val, width=0.8)

    plt.title("Token-Level Accuracy by Command Length", fontsize=16)
    plt.xlabel("Command Length (in words)", fontsize=14)
    plt.ylabel("Accuracy on New Commands (%)", fontsize=14)

    plt.grid(axis="y", linestyle="-", color="lightgray")
    plt.ylim(0, 100)

    plt.xticks(positions, x_val)

    plt.show()
    plt.savefig("ex2_token_acc_by_command_length.png")


def plot_ex2_seqAccCommandLength(x_val, y_val):
    plt.figure(figsize=(10, 6))
    positions = np.arange(len(x_val))

    plt.bar(positions, y_val, width=0.8)

    plt.title("Sequence-Level Accuracy by Command Length", fontsize=16)
    plt.xlabel("Command Length (in words)", fontsize=14)
    plt.ylabel("Accuracy on New Commands (%)", fontsize=14)

    plt.grid(axis="y", linestyle="-", color="lightgray")
    plt.ylim(0, 100)

    plt.xticks(positions, x_val)

    plt.show()
    plt.savefig("ex2_seq_acc_by_command_length.png")


def plot_ex3_tokenLevelAcc(x_val, y_val):
    plt.figure(figsize=(10, 6))
    positions = np.arange(len(x_val))

    plt.bar(positions, y_val, width=0.8)

    plt.title("Token-Level Accuracy", fontsize=16)
    plt.xlabel("Number of Composed Commands Used For Training", fontsize=14)
    plt.ylabel("Accuracy on new commands (%)", fontsize=14)

    plt.grid(axis="y", linestyle="-", color="lightgray")

    plt.ylim(0, 100)
    plt.xticks(positions, x_val)

    plt.tight_layout()
    plt.show()


def plot_ex3_sequenceLevelAcc(x_val, y_val):
    plt.figure(figsize=(10, 6))
    positions = np.arange(len(x_val))

    plt.bar(positions, y_val, width=0.8)

    plt.title("Sequence-Level Accuracy", fontsize=16)
    plt.xlabel("Number of Composed Commands Used For Training", fontsize=14)
    plt.ylabel("Accuracy on new commands (%)", fontsize=14)

    plt.grid(axis="y", linestyle="-", color="lightgray")

    plt.ylim(0, 100)
    plt.xticks(positions, x_val)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    commands = [
        ["jump", "around right"],
        ["walk", "opposite", "left"],
        ["run", "twice"],
        ["walk", "around", "left"],
        ["jump", "opposite", "right"],
    ]
    gold_actions = [
        [
            "JUMP",
            "TURN_RIGHT",
            "JUMP",
            "TURN_RIGHT",
            "JUMP",
            "TURN_RIGHT",
            "JUMP",
            "TURN_RIGHT",
        ],
        ["WALK", "TURN_LEFT", "TURN_LEFT", "WALK"],
        ["RUN", "RUN"],
        [
            "WALK",
            "TURN_LEFT",
            "WALK",
            "TURN_LEFT",
            "WALK",
            "TURN_LEFT",
            "WALK",
            "TURN_LEFT",
        ],
        ["JUMP", "TURN_RIGHT", "JUMP", "TURN_RIGHT"],
    ]
    pred_actions = [
        [
            "JUMP",
            "TURN_RIGHT",
            "JUMP",
            "TURN_RIGHT",
            "RUN",
            "TURN_RIGHT",
            "JUMP",
            "TURN_RIGHT",
        ],
        ["WALK", "TURN_LEFT", "TURN_LEFT", "WALK"],
        ["RUN", "RUN"],
        [
            "WALK",
            "TURN_LEFT",
            "WALK",
            "TURN_LEFT",
            "WALK",
            "TURN_LEFT",
            "WALK",
            "TURN_LEFT",
        ],
        ["JUMP", "TURN_RIGHT", "JUMP", "TURN_RIGHT"],
    ]
    results = compute_accuracies_ex2(commands, gold_actions, pred_actions)
    plot_ex2_tokenAccSeqLength(
        results["action_lengths"], results["token_accuracy_by_action_length"]
    )
    plot_ex2_tokenAccCommandLength(
        results["command_lengths"], results["token_accuracy_by_command_length"]
    )
    # example_x = [24, 25, 26, 27, 28, 30, 32, 33, 36, 40, 48]
    # example_y = [92, 78, 70, 63, 72, 66, 64, 58, 66, 50, 47]
    # # plot_ex2_acSeqLength(example_x, example_y)

    # example_x2 = [4, 6, 7, 8, 9]
    # example_y2 = [96, 90, 85, 75, 60]
    # # plot_ex2_accCommandLength(example_x2, example_y2)

    # example_x3 = [0, 1, 2, 4, 8, 16, 32]
    # example_y3 = [50, 60, 70, 80, 85, 90, 95]
    # # plot_ex3_tokenLevelAcc(example_x3, example_y3)

    # example_x4 = [0, 1, 2, 4, 8, 16, 32]
    # example_y4 = [20, 30, 40, 50, 60, 70, 80]
    # plot_ex3_sequenceLevelAcc(example_x4, example_y4)
