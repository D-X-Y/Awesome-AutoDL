#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2022.01 #
########################################################################################
# python -m awesome_autodl.bins.list_email                                             #
########################################################################################
import argparse
from collections import OrderedDict, defaultdict
from awesome_autodl import autodl_topic2papers


def generate_email(author, counts):
    message = (
        f"Dear {author},"
        + "\n\n"
        + "We at the UTS' Complex Adaptive Systems Lab, part of the Data Science Institute in Sydney, Australia, have recently compiled a comprehensive conceptual review of automated deep learning (AutoDL), see initial version at https://arxiv.org/abs/2112.09245."
        + "\n\n"
        + "The intent of this review is to establish a broader snapshot of the AutoDL field beyond just neural architecture search (NAS), as well as motivate a discussion on how best to evaluate the quality of AutoDL research."
        + "\n\n"
    )
    if counts >= 3:
        message += f"In compiling this review, we have found more than {counts} of your works (referenced in the review) to be significant for informing some of the content, and we hope you find the paper of interest."
    else:
        message += "In compiling this review, we have found your work (referenced in the review) to be significant for informing some of the content, and we hope you find the paper of interest."
    message += (
        "\n\n"
        + "We thus welcome any feedback on the accuracy of our coverage - particularly with regard to your referenced contributions - and invite a broader discussion, if desired."
        + "\n\n"
        + "If you have moved on from this research area or are otherwise uninterested, we apologise for any inconvenience."
        + "\n\n"
        + "Best regards,"
        + "\n"
        + "Xuanyi Dong, David J. Kedziora, Kaska Musial and Bogdan Gabrys"
    )
    return message


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Analysis the AutoDL papers.")
    parser.add_argument(
        "--output_file",
        type=str,
        help="The path to save the final directory.",
    )
    args = parser.parse_args()

    author_email_title = []
    topic2papers = autodl_topic2papers()
    for topic, papers in topic2papers.items():
        # print(f'Collect {len(papers)} papers for "{topic}"')
        for paper in papers:
            if not paper.discussed:
                continue
            for author, email in paper.author_email.items():
                # print(f"{author:25s}, {email:15s} : {paper.title}")
                assert email is not None, "f{paper} has a None value for email"
                author_email_title.append((author, email.lower(), paper.title))
    # print(f"There are {len(author_email_title)} items in total.")

    author2email = OrderedDict()
    author2counts = defaultdict(lambda: 0)
    for author, email, title in author_email_title:
        if author in author2email:
            assert (
                author2email[author] == email
            ), f"{author} : {author2email[author]} vs {email}"
        author2email[author] = email
        author2counts[author] += 1
    # print(f"There are {len(author2email)} unique authors.")

    for author, email in author2email.items():
        message = generate_email(author, author2counts[author])
        print(f"\n\nemail:to:{email}")
        print(f"message:\n{message}")
