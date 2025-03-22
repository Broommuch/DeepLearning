## 首先新建本地仓库



```
git init
```

## 然后创建本地分支并关联到远程已有仓库分支

**如果本地分支名与远程分支名相同**（例如远程分支 `dev`，本地也命名 `dev`）：

```bash
git switch -c dev origin/dev
```

**如果本地分支名与远程分支名不同**（例如远程分支 `feature/login`，本地命名为 `login`）：

```bash
git switch -c login origin/feature/login
```



方法二：

```
1.查看一下本地分支

    git branch;

    查看本地和远程的所有分支

    git branch -a

2.新建一个本地的分支

    git branch -b newbranch   //这个命令是新建一个分支，并切换到该分支上去

    （git branch newbranch;     git checkout newbranch）这两个命令合起来等同于上面的一个命令

3.新建一个远程分支（同名字的远程分支）

    git push origin newbranch:newbranch   //创建了一个远程分支名字叫 newbranch

4.把本地的新分支，和远程的新分支关联

    git push --set-upstream origin newbranch
```