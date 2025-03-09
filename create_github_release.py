import requests
import json
import argparse
import os

def create_github_release(token, repo_owner, repo_name, tag_name, name, body, draft=False, prerelease=False):
    """
    Create a new release on GitHub using the GitHub API
    
    Args:
        token (str): GitHub Personal Access Token
        repo_owner (str): Repository owner/organization
        repo_name (str): Repository name
        tag_name (str): Tag name for the release
        name (str): Release title
        body (str): Release description
        draft (bool): Whether the release is a draft
        prerelease (bool): Whether the release is a pre-release
    
    Returns:
        dict: Response from GitHub API
    """
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases"
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    data = {
        "tag_name": tag_name,
        "name": name,
        "body": body,
        "draft": draft,
        "prerelease": prerelease
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 201:
        print(f"✅ Release created successfully: {response.json()['html_url']}")
        return response.json()
    else:
        print(f"❌ Failed to create release: {response.status_code}")
        print(response.text)
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a new release on GitHub")
    parser.add_argument("--token", help="GitHub Personal Access Token", required=False)
    parser.add_argument("--owner", help="Repository owner/organization", default="notcaliper")
    parser.add_argument("--repo", help="Repository name", default="HandGaze")
    parser.add_argument("--tag", help="Tag name for the release", default="v1.0.0")
    parser.add_argument("--name", help="Release title", default="HandGaze v1.0.0 - Initial Stable Release")
    parser.add_argument("--body-file", help="File containing release description", default="RELEASE_NOTES.md")
    parser.add_argument("--draft", help="Whether the release is a draft", action="store_true")
    parser.add_argument("--prerelease", help="Whether the release is a pre-release", action="store_true")
    
    args = parser.parse_args()
    
    # Get token from environment variable if not provided as argument
    token = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GitHub token not provided. Use --token argument or set GITHUB_TOKEN environment variable.")
        exit(1)
    
    # Read release body from file
    try:
        with open(args.body_file, "r") as file:
            body = file.read()
    except FileNotFoundError:
        print(f"Error: Body file '{args.body_file}' not found")
        exit(1)
    
    # Create the release
    create_github_release(
        token=token,
        repo_owner=args.owner,
        repo_name=args.repo,
        tag_name=args.tag,
        name=args.name,
        body=body,
        draft=args.draft,
        prerelease=args.prerelease
    ) 