import numpy as np


def find_episode(frame_idx: int, episode_ends: np.ndarray) -> int:
    """
    Find which episode the given frame_idx belongs to.

    Parameters
    ----------
    frame_idx : int
        The index of the frame.
    episode_ends : np.ndarray
        The array defining the end indices of episodes.

    Returns
    -------
    episode_number : int
        The zero-based episode number that frame_idx belongs to, or -1 if not found.
    """
    start = 0
    for epi_idx, end in enumerate(episode_ends):
        if start <= frame_idx < end:
            return epi_idx
        start = end
    return -1


def fill_isolated_frames(valid_mask: np.ndarray, episode_ends: np.ndarray) -> np.ndarray:
    """
    Adjust valid_mask so that no frames are isolated within the same episode.
    An "isolated" frame is defined as a single True surrounded by Falses on both sides: pattern F,T,F.
    We will turn neighbors to True as well, but only if they are in the same episode.

    Parameters
    ----------
    valid_mask : np.ndarray(bool)
        The original valid_mask.
    episode_ends : np.ndarray(int)
        The array defining the end indices of episodes.

    Returns
    -------
    adjusted_valid_mask : np.ndarray(bool)
        New valid_mask with no isolated True frames that cause cross-episode bridging.
    """
    adjusted_mask = valid_mask.copy()
    length = len(adjusted_mask)

    for i in range(1, length - 1):
        # Check for isolated True: pattern F,T,F
        if valid_mask[i] and not valid_mask[i - 1] and not valid_mask[i + 1]:
            # Find which episode frame i belongs to
            epi_i = find_episode(i, episode_ends)
            if epi_i == -1:
                continue

            # Check episodes for neighbors
            epi_i_minus_1 = find_episode(i - 1, episode_ends)
            epi_i_plus_1 = find_episode(i + 1, episode_ends)

            # Make neighbors valid only if they are in the same episode
            if epi_i_minus_1 == epi_i:
                adjusted_mask[i - 1] = True
            if epi_i_plus_1 == epi_i:
                adjusted_mask[i + 1] = True

    return adjusted_mask


def compute_new_episode_ends(original_episode_ends: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Compute new episode ends after filtering frames.

    Parameters
    ----------
    original_episode_ends : np.ndarray
        Array of integers defining the end indices of episodes in the original unfiltered data.
    valid_mask : np.ndarray (bool)
        Boolean array of the same length as the original data, where True indicates the frame
        is kept after filtering and False indicates it is removed.

    Returns
    -------
    new_episode_ends : np.ndarray
        The episode_ends arrays in the filtered index space after filtering.
    """
    filtered_indices = np.where(valid_mask)[0]

    if len(filtered_indices) == 0:
        # No frames left, return empty episode ends
        return np.array([], dtype=np.int64)

    # Precompute a mapping from old (original) indices to new (filtered) indices
    old_to_new_idx = {old_i: new_i for new_i, old_i in enumerate(filtered_indices)}

    new_episode_ends = []
    start_idx = 0

    for epi_end in original_episode_ends:
        epi_start = start_idx
        start_idx = epi_end

        # Frames in this original episode that survived filtering
        epi_filtered_frames = filtered_indices[(filtered_indices >= epi_start) & (filtered_indices < epi_end)]

        if len(epi_filtered_frames) == 0:
            # No frames left from this episode
            continue

        # Identify contiguous runs in epi_filtered_frames
        prev_frame = epi_filtered_frames[0]
        for frm in epi_filtered_frames[1:]:
            # Check if there's a continuity break
            if frm != prev_frame + 1:
                # Close off the previous run
                new_episode_ends.append(old_to_new_idx[prev_frame] + 1)
            prev_frame = frm

        # Close off the last run for this episode
        new_episode_ends.append(old_to_new_idx[prev_frame] + 1)

    return np.array(new_episode_ends, dtype=np.int64)


# ------------------------------------------
# TEST FUNCTIONS
# ------------------------------------------

def test_remove_isolated_frames_with_episodes():
    # Same test from before
    original_ends = np.array([5, 10], dtype=np.int64)  # 2 episodes
    original_mask = np.array([False, True, False, False, False, True, False, False, True, False])
    adjusted_mask = fill_isolated_frames(original_mask, original_ends)
    expected_mask = np.array([True, True, True, False, False, True, True, True, True, True])
    assert np.array_equal(adjusted_mask, expected_mask), f"Expected {expected_mask}, got {adjusted_mask}"
    print("test_remove_isolated_frames_with_episodes passed!")


def test_compute_new_episode_ends_multiple_episodes():
    # Now let's consider multiple episodes, e.g.
    # original_ends = [3,6,10]
    # Episodes:
    #   E0: [0..2], E1: [3..5], E2: [6..9]
    #
    # Let's make a valid_mask with isolated frames across episodes
    # Index:  0   1   2 | 3   4   5 | 6   7   8   9
    # Mask:   T   F   T | F   T   F | F   T   F   T
    #
    # After remove_isolated_frames with episode constraints:
    # Check E0: frames=0(T),1(F),2(T)
    # isolated at i=2? neighbors:1(F same epi),3(F next epi) -> make 1=T (not 3 because different epi)
    # E0 final: [T,T,T]
    #
    # E1: frames=3(F),4(T),5(F)
    # isolated at i=4? neighbors:3(F same epi?), let's see episodes:
    #   frame 4 in E1
    #   frame 3 also in E1 (since E1=[3..5]), so 3=T
    #   frame 5 also in E1, so 5=T
    # E1 final: [T,T,T]
    #
    # E2: frames=6(F),7(T),8(F),9(T)
    # isolated at i=7? neighbors:6(F in E2?), yes E2=[6..9], so 6=T; 8(F in E2?), yes 8=T
    # After fix for i=7: E2: [F,T,F,T] -> [T,T,T,T]
    # Now re-check i=9, it's not isolated anymore (since 8=T)
    # E2 final: [T,T,T,T]
    #
    # final mask after adjustments:
    # E0: [0,1,2] all T
    # E1: [3,4,5] all T
    # E2: [6,7,8,9] all T
    #
    # final: T,T,T | T,T,T | T,T,T,T
    # filtered_indices = [0,1,2,3,4,5,6,7,8,9] unchanged
    # Runs:
    # E0: single run 0..2 => end at 3
    # E1: single run 3..5 => end at 6
    # E2: single run 6..9 => end at 10
    #
    # expected: [3,6,10]

    original_ends = np.array([3, 6, 10], dtype=np.int64)
    original_mask = np.array([True, False, True, False, True, False, False, True, False, True])
    adjusted_mask = fill_isolated_frames(original_mask, original_ends)
    new_ends = compute_new_episode_ends(original_ends, adjusted_mask)
    expected_new_ends = np.array([3, 6, 10], dtype=np.int64)

    assert np.array_equal(new_ends, expected_new_ends), f"Expected {expected_new_ends}, got {new_ends}"
    print("test_compute_new_episode_ends_multiple_episodes passed!")


if __name__ == "__main__":
    # Run tests
    test_remove_isolated_frames_with_episodes()
    test_compute_new_episode_ends_multiple_episodes()
