# Codes running on Ocean.pangeo Platform for Analyzing CMIP6 Data


## Introduction
This repository shares some basic codes for analyzing CMIP6 Data output on ocean.pangeo platform.

This document is a simple toturial for using the `git` Version Control System to modify and contribute to this repository.

## Cloning the repository

To clone a complete copy of this repository to your pangeo folder, use `git clone`:

<pre><code>git clone https://github.com/Marinov-Ocean-Group/CMIP6-Codes.git</code></pre>

Then you can use the codes in the local repository on google platform, and also make modifies to the codes.

## Modifying the code

1. **Create an issue**

Before beggining the work, an issue should be created on Github [issue tracker](https://github.com/Marinov-Ocean-Group/CMIP6-Codes/issues). This allows us to keep track of what need to be done, when and why. The issue can be asigned to whoever can help with the modification.

2. **Checkout branches**

The master branch is the "default" branch. To avoid conflict among mutiple modifications, you may need to use other branches for development and merge them back to the master branch upon completion. 
    
This can be done by using `git checkout`:
    <pre><code>git checkout -b BRANCH-NAME</code></pre>
    
where BRANCH-NAME is the name of your custom branch. You can now go ahead and modify any code you like.

3. **Add and commit the changes**

Use `git status` to check what change have been done in the local repo folder.

To add changes to the "HEAD" file, use `git add <FILENAME>`
    
To commit the changes, use `git commit -m "Commit Message"`
        
4. **Push changes back to GitHub** 

**Note**: You may need to set your GitHub username first:
<pre><code>git config --global user.name "Username"</code></pre>
<pre><code>git config --global user.email "email@example.com"</code></pre>
   
After committing the changes and setting the username, the local repo can be upload to the remote repo by using git push:

<pre><code>git push origin BRANCH-NAME</code></pre>
    

5. **Create a merge request upon completion** 

On GitHub make a merge request between your branch and the default. 

## Deleting old branches

To delete an old, no-longer used branch both locally and remotely use:

<pre><code>git del BRANCH-NAME</code></pre>
   

## Additional Notes

