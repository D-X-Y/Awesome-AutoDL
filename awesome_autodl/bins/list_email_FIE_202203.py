#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2022.03 #
########################################################################################
# python -m awesome_autodl.bins.list_email_FIE_202203                                  #
########################################################################################
import argparse
from collections import OrderedDict, defaultdict
from awesome_autodl import autodl_topic2papers
from awesome_autodl.utils import email_old_to_new_202203
from awesome_autodl.bins.list_email import generate_email


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
        if (
            email in email_old_to_new_202203
            and email_old_to_new_202203[email] is not None
        ):
            author2email[author] = email
            author2counts[author] += 1
    print(
        f"During fixing the invalid email issue, there are {len(author2email)} unique authors."
    )

    for author, email in author2email.items():
        message = generate_email(author, author2counts[author])
        print(f"\n\nemail:to:{email}")
        print(f"message:\n{message}")
