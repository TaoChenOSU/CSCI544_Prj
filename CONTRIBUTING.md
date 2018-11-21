# Contributing to this project

## Python style

### Python3
Please use python3 for our project.

### Indents
Please use 4-space instead of tabs for indents. 

## Git
### Development
Since this is a private repository, you can't fork it to your own. So to make a better collaboration and version control, you should create a branch of your own in github.

```bash
git clone https://github.com/TaoChenOSU/CSCI544_Prj.git 
git checkout -b branch(#your_branch)
```

And when you finished a function, add the modified files and a commit message. So that a new log will be shown in your branch log tree.

Then next thing is to push the code to a remote branch, then create a pull request so that you can merge your code into the master branch. 

```bash
# use git status to see which files you need to add
git status
# use git add to add the files 
git add (#some_file)
# use git commit -m to quickly commit a msg
git commit -m "Some updates"
```

Here is a useful resource for [good commit messages](https://chris.beams.io/posts/git-commit/).

If you already committed a message in your branch, but you want to add some small changes to your code. You can use `git commit --amend`. It will open an editor in your terminal and you can change the previous message. This provides a clean log tree your branch, avoiding unnessary verbose messages.

### Merge
Now everything is updated in your local branch. We need to push it to the remote branch.

```bash
git fetch 
git rebase origin/master
git push -f origin (#your_branch)
```

Use the above three lines. And then create a pull request on github.

You can probably merge the pull requests yourself as we are not dealing with a very serious project. But remember using `Squash and merge` when you merge the pr. This is important because if you choose `Create a merge commit`, it will create two commits in the master branch, which is really ugly and hard to track.