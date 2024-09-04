# loop from 0 to 49 (including 49), run `python test_aloha_IK_speed.py -i i`
for i in {0..49}
do
    python test_aloha_IK_speed.py -i $i
done

# for i in {0..10}
# do
#     python test_aloha_IK_speed.py -i $i -t sim_insertion_scripted
# done
