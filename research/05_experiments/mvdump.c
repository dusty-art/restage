// mvdump: decode a video with FFmpeg and dump codec motion vectors + frame types.
// Written for the 05_unconventional_ideas research note (codec-as-free-computation test).
// Build: clang -O2 mvdump.c -I/opt/homebrew/include -L/opt/homebrew/lib -lavformat -lavcodec -lavutil -o mvdump
// Usage: ./mvdump in.mp4 out.bin   (out.bin = "-" to skip writing; prints per-frame summary to stdout)
// Binary record per MV: int32 frame, int32 source, int16 w, int16 h, int16 dst_x, int16 dst_y,
//                       float32 dx, float32 dy   (dx,dy = src - dst in pixels)
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/motion_vector.h>
#include <stdio.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

int main(int argc, char **argv){
  if(argc<3){ fprintf(stderr,"usage: %s in out.bin|-\n",argv[0]); return 1; }
  AVFormatContext *fmt=NULL;
  if(avformat_open_input(&fmt,argv[1],NULL,NULL)<0){ fprintf(stderr,"open fail\n"); return 1; }
  avformat_find_stream_info(fmt,NULL);
  const AVCodec *dec=NULL;
  int si=av_find_best_stream(fmt,AVMEDIA_TYPE_VIDEO,-1,-1,&dec,0);
  if(si<0){ fprintf(stderr,"no video\n"); return 1; }
  AVCodecContext *ctx=avcodec_alloc_context3(dec);
  avcodec_parameters_to_context(ctx,fmt->streams[si]->codecpar);
  AVDictionary *opts=NULL; av_dict_set(&opts,"flags2","+export_mvs",0);
  ctx->thread_count=1; // MV side data is most reliable single-threaded
  if(avcodec_open2(ctx,dec,&opts)<0){ fprintf(stderr,"codec open fail\n"); return 1; }
  FILE *out = strcmp(argv[2],"-")? fopen(argv[2],"wb"): NULL;
  AVPacket *pkt=av_packet_alloc(); AVFrame *fr=av_frame_alloc();
  int fi=0; long total=0; double t0=now();
  printf("codec=%s\n", dec->name);
  while(1){
    int r=av_read_frame(fmt,pkt);
    if(r<0){ avcodec_send_packet(ctx,NULL); } else if(pkt->stream_index!=si){ av_packet_unref(pkt); continue; } else { avcodec_send_packet(ctx,pkt); }
    while(avcodec_receive_frame(ctx,fr)==0){
      AVFrameSideData *sd=av_frame_get_side_data(fr,AV_FRAME_DATA_MOTION_VECTORS);
      int n = sd? sd->size/sizeof(AVMotionVector): 0;
      long area=0;
      if(sd){ const AVMotionVector *mv=(const AVMotionVector*)sd->data;
        for(int i=0;i<n;i++){ area += mv[i].w*mv[i].h;
          if(out){ int32_t a[2]={fi,mv[i].source}; int16_t b[4]={mv[i].w,mv[i].h,mv[i].dst_x,mv[i].dst_y};
            float c[2]={ (float)mv[i].motion_x/mv[i].motion_scale, (float)mv[i].motion_y/mv[i].motion_scale };
            fwrite(a,4,2,out); fwrite(b,2,4,out); fwrite(c,4,2,out);} } }
      // print: frame idx, picture type, number of MVs, fraction of frame area covered by inter blocks
      printf("F %d %c %d %.4f\n", fi, av_get_picture_type_char(fr->pict_type), n, (double)area/(fr->width*fr->height));
      total+=n; fi++; av_frame_unref(fr);
    }
    if(r<0) break; av_packet_unref(pkt);
  }
  double dt=now()-t0;
  printf("frames=%d mvs=%ld seconds=%.3f fps=%.1f\n",fi,total,dt,fi/dt);
  if(out) fclose(out);
  return 0;
}
