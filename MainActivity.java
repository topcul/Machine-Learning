package com.example.sr.progressbarexample;

import android.os.Handler;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {
   private ProgressBar progressBar;
   private TextView textView;
   private Handler handler=new Handler();
   private Button baslat;
   private Button durdur;
   private Button sifirla;
   private ProgressBar halka;
   private int progressStatus=0;
   private int progressStatus2=0;
   private boolean suspended=false;//Durdur butonuna basıldığında bu değeri true yapılacak.
   private boolean stopped=false;//Sıfırla butonuna basıldığında bu değeri true yapılacak.

   //Uygulama açıldığında çalıştırılan metod.
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        progressBar=(ProgressBar) findViewById(R.id.progressBar);
        textView=(TextView)findViewById(R.id.textView3);
        textView=(TextView)findViewById(R.id.textView4);
        baslat=(Button)findViewById(R.id.baslat);
        durdur=(Button)findViewById(R.id.durdur);
        sifirla=(Button)findViewById(R.id.sifirla);
        halka=(ProgressBar)findViewById(R.id.progressBar2);

        halka.setIndeterminate(true);
        halka.setMax(60);
        progressBar.setMax(60);//Progressbar'ın max. değeri
        progressBar.setIndeterminate(false);//Progressbarın tekrar eden bir animasyonla çalışması engellenir.

        initValues();
        baslat.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                setKronometre().start();
                stopped=false;
                baslat.setEnabled(false);
                durdur.setEnabled(true);
                sifirla.setEnabled(true);
            }
        });
        durdur.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(suspended){
                    suspended=false;
                    durdur.setText("Durdur");}
                    else{
                    suspended=true;
                    durdur.setText("Devam Et");
                }
            }
        });
        sifirla.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                stopped=true;
                initValues();
            }
        });
    }
    private void initValues(){  //Başlangıç değerleri set edilir.
        progressStatus=0;
        progressStatus2=0;
        progressBar.setProgress(progressStatus);
        halka.setProgress(progressStatus2);
        textView.setText("0sn/60sn");
        baslat.setEnabled(true);
        durdur.setEnabled(false);
        sifirla.setEnabled(false);
        durdur.setText("Durdur");
        suspended=false;
    }
    private Thread setKronometre(){
        return new Thread(new Runnable() {
            @Override
            public void run() {
                while(progressStatus<60 & progressStatus2<30){
                    while(suspended){  //Eğer kronometre durdurulduysa bekle.
                        try{
                            Thread.sleep(300);
                        }
                        catch(InterruptedException e){
                            e.printStackTrace();
                        }
                    } if(stopped) //Eğer kronometre sıfırlandıysa işlemi sonlandır.
                        break;
                    progressStatus+=1;
                    progressStatus2+=2;
                    //yeni değeri ekranda göster ve progressBar' set et.
                    handler.post(new Runnable() {
                        @Override
                        public void run() {
                            progressBar.setProgress(progressStatus);
                            halka.setProgress(progressStatus2);
                            textView.setText(progressStatus+ "sn/"+ progressBar.getMax()+"sn");
                        }
                    });
                    try{
                        //1 sn için uyut.
                        //Süreci yavaş göstermek için
                        Thread.sleep(1000);
                    }
                    catch(InterruptedException e){
                        e.printStackTrace();
                    }
                }

            }
        });
    }
}
