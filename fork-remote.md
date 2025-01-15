<!--
 * @Author: FutureMeng be_loving@163.com
 * @Date: 2025-01-15 22:45:17
 * @LastEditors: FutureMeng be_loving@163.com
 * @LastEditTime: 2025-01-15 22:45:53
 * @FilePath: \MinerU\fork-remote.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
1. 查看是否添加了更新源
   ```
   git remote -v
   ```
2. 添加更新源，本项目fork自yeszao/dnmp
   ```
   git remote add upstream https://github.com/opendatalab/MinerU.git
   ```
3. 从源更新
   ```
   git fetch upstream
   ```
4. 合并源的分支
   ```
   git merge upstream/master
   ```

5. 推送
   ```
   git push
   ```