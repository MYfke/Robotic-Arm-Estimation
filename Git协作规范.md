# Git协作开发规范
一个庞大的工程的建立，需要许多人的一起开发，但是每个人又都有着不同的编程习惯，所以我们需要一个统一的规范，
来确保项目结构清晰，下文，将具体列出开发规范。


## 1.经常性更新本地库
项目的远程库时多人协作开发，因此代码会经常性产生更新，因此，一个良好的习惯就是，经常性拉取远程库，对本地库进行更新


## 2.千万不要直接在main分支上直接提交代码
使用Git过程中，必须通过创建分支进行开发，坚决禁止在主干分支上直接开发。
例如，你要实现一个功能，先新建一个分支，再在分支上进行提交，当分支上的功能已经实现后，则可以将分支合并到主分支。

## 3.合并main分支时，一定要更新本地库
如果你的main分支在不是最新的情况下进行提交操作，那么在之后提交远程库时，会发生代码冲突，因此为了避免这种情况，
在将自己的开发分支合并到主分支时，需要拉取(pull)远程库，来更新本地库，之后再进行合并操作。

## 4.及时推送远程库
当你的本地代码已经实现了某个功能，且已经合并到main分支后，此时需要将代码及时推送(push)至远程库，
来确保其他人能收到你的代码。

## 5.推送远程库时，建议只推送main分支
这一条不做过多要求，假如你的某一开发分支还没有合并到main分支上，那就说明这个分支还没有完工，
那么此时将这个分支提交到远程库也没什么用。除非，这个分支的功能你实现不了，此时可以将分支提交，
然后别人再拉取你的分支帮你完成。

## 6.对main使用合并(merge)操作,不要使用变基(rebase)操作
main分支上需要展示多人的提交日志，因此合并(merge)操作对于多人法人提交更清晰明了
但是对于自己的开发分支，则推荐使用变基(rebase)操作，因为可以更清楚体现自己的开发过程，
例如，自己有两个开发分支，那么就可以将这两个分支进行变基，变为一个分支，之后再将分支合并(merge)到main上。

## 7.Git的提交说明越详细越好
你得把你的提交写明白呀，不然别人怎么知道你提交的是什么东西

## 8.一次commit只提交一个文件
即使你更改了许多文件的代码，那也要一个文件一个文件的提交，并对每个提交写清楚说明，
这样子既能保证commit历史干净，也能保证代码回滚时少出差错。
