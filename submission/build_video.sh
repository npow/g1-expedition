#!/bin/zsh
set -euo pipefail

cd "${0:A:h}"

mkdir -p build/segments build/audio build/contact_sheets

python3 build_assets.py

enc=(-c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -r 30 -an)

ffmpeg -hide_banner -loglevel error -loop 1 -i build/cards/title.png -t 4 \
  -vf "fps=30,fade=t=out:st=3.65:d=0.35" \
  "${enc[@]}" build/segments/01_title.mp4 -y

ffmpeg -hide_banner -loglevel error \
  -ss 2 -t 6 -i source_videos/g1_self_arrest_diverse_suite.mp4 \
  -i source_videos/g1_fixed_line_fall_recovery.mp4 \
  -stream_loop -1 -i source_videos/g1_rappel_footplant_full_preview.mp4 \
  -ss 4 -t 6 -i source_videos/tree.mp4 \
  -loop 1 -i build/cards/overlay_montage.png \
  -filter_complex "[0:v]setpts=PTS-STARTPTS,scale=960:540:force_original_aspect_ratio=increase,crop=960:540,fps=30[a];[1:v]setpts=PTS-STARTPTS,scale=960:540:force_original_aspect_ratio=increase,crop=960:540,fps=30[b];[2:v]trim=duration=6,setpts=PTS-STARTPTS,scale=960:540:force_original_aspect_ratio=increase,crop=960:540,fps=30[c];[3:v]setpts=PTS-STARTPTS,scale=960:540:force_original_aspect_ratio=increase,crop=960:540,fps=30[d];[a][b][c][d]xstack=inputs=4:layout=0_0|960_0|0_540|960_540[grid];[grid][4:v]overlay=0:0:shortest=1,fade=t=in:st=0:d=0.3,fade=t=out:st=5.65:d=0.35[out]" \
  -map "[out]" -t 6 "${enc[@]}" build/segments/02_montage.mp4 -y

ffmpeg -hide_banner -loglevel error -i source_videos/g1_self_arrest_diverse_suite.mp4 -loop 1 -i build/cards/overlay_self.png \
  -filter_complex "[0:v]trim=start=0:end=27.5,setpts=PTS-STARTPTS,fps=30[base];[base][1:v]overlay=0:0:shortest=1,fade=t=in:st=0:d=0.3,fade=t=out:st=27.15:d=0.35[out]" \
  -map "[out]" -t 27.5 "${enc[@]}" build/segments/04_self.mp4 -y

ffmpeg -hide_banner -loglevel error -i source_videos/slip_recovery_final.mp4 -i source_videos/fall_recovery.mp4 -loop 1 -i build/cards/overlay_fixed.png \
  -filter_complex "[0:v]trim=start=0:end=11.48,setpts=PTS-STARTPTS,scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps=30[a];[1:v]trim=start=0:end=12.09,setpts=PTS-STARTPTS,scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps=30[b];[a][b]concat=n=2:v=1:a=0[joined];[joined]tpad=stop_mode=clone:stop_duration=0.43[base];[base][2:v]overlay=0:0:shortest=1,fade=t=in:st=0:d=0.3,fade=t=out:st=23.65:d=0.35[out]" \
  -map "[out]" -t 24 "${enc[@]}" build/segments/05_fixed.mp4 -y

ffmpeg -hide_banner -loglevel error -i source_videos/g1_rappel_long.mp4 -i source_videos/g1_rappel_footplant_full_preview.mp4 -loop 1 -i build/cards/overlay_rappel.png \
  -filter_complex "[0:v]trim=start=0:end=17.68,setpts=PTS-STARTPTS,scale=1920:1080,fps=30[a];[1:v]trim=start=0.4:end=4.72,setpts=PTS-STARTPTS,scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps=30[b];[a][b]concat=n=2:v=1:a=0[base];[base][2:v]overlay=0:0:shortest=1,fade=t=in:st=0:d=0.3,fade=t=out:st=21.65:d=0.35[out]" \
  -map "[out]" -t 22 "${enc[@]}" build/segments/06_rappel.mp4 -y

ffmpeg -hide_banner -loglevel error -i source_videos/tree.mp4 -i source_videos/lifting_log.mp4 -i source_videos/failure.mp4 \
  -loop 1 -i build/cards/overlay_team.png -loop 1 -i build/cards/overlay_voice_1.png \
  -loop 1 -i build/cards/overlay_voice_2.png -loop 1 -i build/cards/overlay_voice_3.png \
  -filter_complex "[0:v]trim=start=33:end=42,setpts=PTS-STARTPTS,scale=1920:1080,fps=30[a];[1:v]trim=start=0:end=14.975,setpts=PTS-STARTPTS,scale=1920:1080,fps=30,tpad=stop_mode=clone:stop_duration=0.025[b];[2:v]trim=start=3.5:end=11.5,setpts=PTS-STARTPTS,scale=1920:1080,fps=30[c];[a][b][c]concat=n=3:v=1:a=0[base];[base][3:v]overlay=0:0[branded];[branded][4:v]overlay=0:0:enable='between(t,14.5,19)'[v1];[v1][5:v]overlay=0:0:enable='between(t,19,24)'[v2];[v2][6:v]overlay=0:0:enable='between(t,24,32)',fade=t=in:st=0:d=0.3,fade=t=out:st=31.65:d=0.35[out]" \
  -map "[out]" -t 32 "${enc[@]}" build/segments/07_team.mp4 -y

ffmpeg -hide_banner -loglevel error -loop 1 -i build/cards/outro.png -t 4.5 \
  -vf "fps=30,fade=t=in:st=0:d=0.35,fade=t=out:st=4.05:d=0.45" \
  "${enc[@]}" build/segments/09_outro.mp4 -y

ffmpeg -hide_banner -loglevel error \
  -i build/segments/01_title.mp4 -i build/segments/02_montage.mp4 \
  -i build/segments/04_self.mp4 -i build/segments/05_fixed.mp4 \
  -i build/segments/06_rappel.mp4 -i build/segments/07_team.mp4 \
  -i build/segments/09_outro.mp4 \
  -filter_complex "[0:v][1:v][2:v][3:v][4:v][5:v][6:v]concat=n=7:v=1:a=0[out]" \
  -map "[out]" -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -r 30 -an build/picture.mp4 -y

zsh -ic "cd '${0:A:h}' && python3 generate_elevenlabs.py"

ffmpeg -hide_banner -loglevel error -f concat -safe 0 -i build/captions/concat.txt \
  -vf "fps=30,format=argb" -c:v qtrle -pix_fmt argb -t 120 build/captions.mov -y

ffmpeg -hide_banner -loglevel error -i build/picture.mp4 -i build/captions.mov \
  -filter_complex "[0:v][1:v]overlay=0:0:shortest=1[out]" -map "[out]" \
  -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -r 30 -an build/picture_captioned.mp4 -y

ffmpeg -hide_banner -loglevel error \
  -i build/audio/narration_1.mp3 -i build/audio/narration_2.mp3 \
  -i build/audio/narration_3.mp3 -i build/audio/narration_4.mp3 \
  -i build/audio/narration_5.mp3 -i build/audio/narration_6.mp3 \
  -i build/audio/narration_7.mp3 \
  -f lavfi -i "aevalsrc=0.025*(sin(2*PI*55*t)+0.55*sin(2*PI*82.41*t)+0.35*sin(2*PI*110*t)):s=48000:d=120" \
  -filter_complex "[0:a]adelay=400,volume=1.0[a0];[1:a]adelay=10200,volume=1.0[a1];[2:a]adelay=37700,volume=1.0[a2];[3:a]adelay=61700,volume=1.0[a3];[4:a]adelay=83700,volume=1.0[a4];[5:a]adelay=98200,volume=1.0[a5];[6:a]adelay=113300,volume=1.0[a6];[7:a]lowpass=f=420,afade=t=in:st=0:d=3,afade=t=out:st=116:d=4,volume=0.10[bed];[a0][a1][a2][a3][a4][a5][a6][bed]amix=inputs=8:duration=longest:normalize=0,loudnorm=I=-16:LRA=8:TP=-1.5,atrim=duration=120[aout]" \
  -map "[aout]" -c:a aac -b:a 192k -ar 48000 -ac 2 build/soundtrack.m4a -y

ffmpeg -hide_banner -loglevel error -i build/picture_captioned.mp4 -i build/soundtrack.m4a \
  -map 0:v:0 -map 1:a:0 -c:v copy -c:a copy -t 120 -movflags +faststart \
  optimus_prime_g1_expedition_2min.mp4 -y

ffmpeg -hide_banner -loglevel error -ss 0 -i optimus_prime_g1_expedition_2min.mp4 \
  -vf "fps=1/12,scale=384:-1,tile=5x2:padding=4:margin=4:color=white" \
  -frames:v 1 build/contact_sheets/final_12s.jpg -y
