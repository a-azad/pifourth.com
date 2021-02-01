RMDIR /Q /S docs\
CD source
hugo
CD ..
git add -A
git commit -m "update"
git push