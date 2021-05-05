list p=18f4520
#include <p18f4520.inc>
CONFIG OSC=HS,PWRT=ON,WDT=OFF,LVP=OFF

#define period_C     0x94  ;每個音的PR2值   
#define period_D     0x84
#define period_E     0x75
#define period_F     0x6E
#define period_G     0x62
#define period_A     0x57
#define period_B     0x4E
#define period_HIGC  0x49

TMR_1     equ .3276     ;每0.1秒中斷一次        
VAL_US    equ .147	;用於delay_100ms	
VAL_MS	  equ .100      ;用於delay_100ms

CBLOCK	0x20	        ;由暫存器位址0x20	
WREG_TEMP	        ;儲存處理器資料重要變數
STATUS_TEMP	        ;儲存處理器資料重要變數
BSR_TEMP                ;儲存處理器資料重要變數
RX_TEMP                 ;儲存從RCREG讀取出來的資料  
KEY_FLAG                ;用來確認有沒有接收過資料
W_TEMP                  ;儲存從WREG讀取出來的資料
count	 		;用於delay_100ms
count_ms                ;用於delay_100ms
PLAY_FLAG               ;用來確認是否開始錄音
RECORD_TEMP             ;儲存PR2的值，需要錄音時則將其值傳送出去
ABC                     ;用於計算錄音與播放的暫存器個數
ABC_2                   ;用於計算次數，以改變LED燈
ABC_3                   ;用於計算次數，以改變LED燈
ENDC

CBLOCK 0x40             ;由暫存器位址0x40
ABC000                  ;給錄音用的110個暫存器
ABC001 
ABC002
ABC003
ABC004
ABC005
ABC006
ABC007
ABC008
ABC009 
ABC010 
ABC011
ABC012
ABC013
ABC014
ABC015
ABC016
ABC017
ABC018
ABC019
ABC020
ABC021 
ABC022
ABC023
ABC024
ABC025
ABC026
ABC027
ABC028
ABC029 
ABC030 
ABC031
ABC032
ABC033
ABC034
ABC035
ABC036
ABC037
ABC038
ABC039
ABC040
ABC041
ABC042
ABC043
ABC044
ABC045
ABC046
ABC047
ABC048
ABC049
ABC050
ABC051
ABC052
ABC053
ABC054
ABC055
ABC056
ABC057
ABC058
ABC059
ABC060
ABC061
ABC062
ABC063
ABC064
ABC065
ABC066
ABC067
ABC068
ABC069
ABC070
ABC071
ABC072
ABC073
ABC074
ABC075
ABC076
ABC077
ABC078
ABC079
ABC080
ABC081
ABC082
ABC083
ABC084
ABC085
ABC086
ABC087
ABC088
ABC089
ABC090
ABC091
ABC092
ABC093
ABC094
ABC095
ABC096
ABC097
ABC098
ABC099
ABC100
ABC101
ABC102
ABC103
ABC104
ABC105
ABC106
ABC107
ABC108
ABC109
ENDC      
      
org 0x00
GOTO INITIAL
;-------------high priority interrupt program----------
org 0x08                ;高優先中斷的程式向量位址
BCF PIR1,TMR1IF         ;清除TIMER1中斷旗標
GOTO HI_ISRS
;-------------low priority interrupt program-----------
org 0x18                ;低優先中斷的程式向量位址
BCF PIR1,RCIF           ;清除RX中斷旗標
GOTO PIANO_KEY
;--------------------Initial program --------------------  
org 0x2A                ;正常執行程式的開始
INITIAL:  
  CALL INIT_IO          ;呼叫所有輸出入腳位設定 
  CALL INIT_CCP         ;呼叫CCP模組設定 
  CALL INIT_TIMER1      ;呼叫TIMER1設定
  CALL INIT_TXRX        ;呼叫UART設定
  CALL INIT_AD          ;呼叫AD轉換設定
  CALL INIT_SETTING     ;呼叫初始化設定
  CALL INIT_ISRS        ;呼叫中斷設定
;------------------Main Program (播放音樂)------------------   
MAIN:
  BTFSS PORTB,0         ;測試SW2有沒有按
  SETF PLAY_FLAG        ;有按的話，將PLAY_TEMP設成0xFF
  BTFSS LATD,0          ;測試LED有沒有亮，有亮代表錄完音了
  GOTO MAIN
  BSF ADCON0,GO         ;開啟類比轉換
  NOP
  BTFSC ADCON0,GO       ;確認轉換完成與否
  BRA $-4
  MOVF ADRESH,W
SW4:                    ;測試SW4有沒有按
  BTFSS STATUS,Z        ;測邏輯運算結果是不是0
  BRA MAIN  
  LFSR FSR1,0x43        ;是的話，令FSR1位址為0x43
  BCF PIE1,TMR1IE       ;關閉TIMER1中斷功能
  MOVLW B'00101111'     ;開啟PWM
  MOVWF CCP1CON
KEY_PLAY:
  MOVFF INDF1,PR2       ;將INDF1傳送至PR2
  INCF FSR1L            ;FSR1L值加1
  INCF ABC              ;ABC值加1
  INCF ABC_3            ;ABC_3值加1
  CALL DUTY_CYCLE       ;給予相對的CCPR1L:CCP1CON<5:4>值
  CALL delay_100ms      ;0.1秒延遲，讓播放音樂和錄音有同樣的解析度
  MOVLW .13             ;檢查ABC_3的值是否為13
  CPFSEQ ABC_3
  BRA KEY_PLAY_2        ;不是的話跳到KEY_PLAY_2    
  BCF STATUS,C           
  RRCF LATD             ;是的話，使LED燈由bit7至bit0一個一個滅
  CLRF ABC_3            ;將ABC_3歸零
  BRA KEY_PLAY 
KEY_PLAY_2:
  MOVLW .105            ;檢查ABC的值是否為105，前後暫存器不要以避免雜訊
  CPFSEQ ABC
  BRA KEY_PLAY            ;不是則跳到KEY_PLAY
  CLRF ABC                ;是的話將ABC歸零
  CLRF ABC_3              ;將ABC_3歸零
  MOVLW B'00100000'       ;關掉PWM
  MOVWF CCP1CON    
  CLRF LATD               ;LED全滅，代表播放完音樂
  BSF PIE1,TMR1IE         ;TIMER中斷重新開啟
  GOTO MAIN
;----------------high priority interrupt program---------------
HI_ISRS:
  BTFSS KEY_FLAG,0        ;測試前0.1秒鐘有沒有接收過訊息
  BRA KEY_END             ;沒有的話，直接跳到KEY_END
  MOVLW B'00101111'       ;有的話，啟動PWM
  MOVWF CCP1CON
KEY_S:                    ;檢查是不是按S
  MOVLW A'S'              
  CPFSEQ RX_TEMP
  BRA KEY_D               ;不是則再繼續確認
  MOVLW period_C          
  MOVWF PR2               ;是的話將相對的值丟入PR2
  MOVWF RECORD_TEMP       ;將值丟入RECORD_TEMP，以利錄音
  CALL DUTY_CYCLE         ;給予相對的CCPR1L:CCP1CON<5:4>值
  BRA KEY_EEND            ;結束檢查，直接跳到KEY_EEND
KEY_D:                    ;檢查是不是按D
  MOVLW A'D'
  CPFSEQ RX_TEMP
  BRA KEY_F
  MOVLW period_D
  MOVWF PR2
  MOVWF RECORD_TEMP
  CALL DUTY_CYCLE
  BRA KEY_EEND
KEY_F:                    ;檢查是不是按F
  MOVLW A'F'
  CPFSEQ RX_TEMP
  BRA KEY_G
  MOVLW period_E
  MOVWF PR2
  MOVWF RECORD_TEMP
  CALL DUTY_CYCLE
  BRA KEY_EEND
KEY_G:                    ;檢查是不是按G
  MOVLW A'G'
  CPFSEQ RX_TEMP
  BRA KEY_H
  MOVLW period_F
  MOVWF PR2
  MOVWF RECORD_TEMP
  CALL DUTY_CYCLE
  BRA KEY_EEND
KEY_H:                    ;檢查是不是按H
  MOVLW A'H'
  CPFSEQ RX_TEMP
  BRA KEY_J
  MOVLW period_G
  MOVWF PR2
  MOVWF RECORD_TEMP
  CALL DUTY_CYCLE
  BRA KEY_EEND
KEY_J:                    ;檢查是不是按J
  MOVLW A'J'
  CPFSEQ RX_TEMP
  BRA KEY_K
  MOVLW period_A
  MOVWF PR2
  MOVWF RECORD_TEMP
  CALL DUTY_CYCLE
  BRA KEY_EEND
KEY_K:                    ;檢查是不是按K
  MOVLW A'K'
  CPFSEQ RX_TEMP
  BRA KEY_L
  MOVLW period_B
  MOVWF PR2
  MOVWF RECORD_TEMP
  CALL DUTY_CYCLE
  BRA KEY_EEND
KEY_L:                    ;檢查是不是按L
  MOVLW A'L'
  CPFSEQ RX_TEMP
  BRA KEY_END             ;不是的話直接跳到KEY_END
  MOVLW period_HIGC
  MOVWF PR2
  MOVWF RECORD_TEMP
  CALL DUTY_CYCLE
  BRA KEY_EEND
KEY_END:                  
  MOVLW B'00100000'       ;若前0.1秒沒接收訊息或檢查無相對的按鍵
  MOVWF CCP1CON           ;則直接關掉PWM
  CLRF RECORD_TEMP        ;將RECORD_TEMP設成0
KEY_EEND:
  BTFSC PLAY_FLAG,1       ;檢查PLAY_TEMP的bit 1是不是1
  CALL KEY_RECORD         ;是的話呼叫KEY_RECORD
  BCF KEY_FLAG,0          ;是0，則將KEY_TEMP的bit 1設為0
  MOVLW (.65536-TMR_1)/.256  ;使TIMER1 0.1秒中斷一次
  MOVWF TMR1H
  MOVLW (.65536-TMR_1)%.256
  MOVWF TMR1L 
  RETFIE FAST             ;中斷結束            
;---------------------------RECORD-----------------------------
KEY_RECORD:
  BTFSS PLAY_FLAG,0       ;檢查PLAY_TEMP的bit0是不是1
  BRA KEY_RECORD_2        ;是0，則直接跳到KEY_RECORD_2
  LFSR FSR0,0x40          ;是1，則初始化FSR0的值
  BCF PLAY_FLAG,0         ;將PLAY_TEMP的bit 0設為0
  CLRF LATD               ;將LATD清除為零
KEY_RECORD_2:             ;開始錄音
  BSF LATD,0              ;將LED的bit 0設為1，代表開始錄音
  MOVFF RECORD_TEMP,INDF0 ;將RECORD_TEMP傳送至INDF0相對應的暫存器
  INCF FSR0L              ;FSR0L值加1
  INCF ABC                ;ABC值加1
  INCF ABC_2              ;ABC_2值加1
  MOVLW .16               ;檢查ABC_2的值是否為16
  CPFSEQ ABC_2            
  BRA KEY_RECORD_3        ;不是的話跳到RECORD_2                      
  CLRF ABC_2              ;將ABC_2歸零
  BSF STATUS,C
  RLCF LATD               ;使LED燈由bit0至bit7一個一個亮
KEY_RECORD_3: 
  MOVLW .110              ;檢查ABC_2的值是否為110
  CPFSEQ ABC
  RETURN                  ;不是則返回
  CLRF ABC                ;是的話將ABC歸零
  CLRF ABC_2              ;將ABC_2歸零
  CLRF PLAY_FLAG          ;將PLAY_TEMP歸零
  SETF LATD               ;將LED設為1
  RETURN
;-------------------設定Duty Cycle為PR2的一半--------------------
DUTY_CYCLE:
  MOVFF PR2,W_TEMP        ;將從暫存器中讀取到的PR2值丟入W_TEMP
  BCF STATUS,C
  RRCF W_TEMP             ;W_TEMP值除以2
  MOVFF W_TEMP,CCPR1L     ;再丟入CCPR1L，使Duty Cycle值為PR2的一半
  RETURN
;---------------------------UART的RX---------------------------
PIANO_KEY:
  MOVFF STATUS,STATUS_TEMP ;儲存處理器資料重要變數
  MOVFF WREG,WREG_TEMP     ;儲存處理器資料重要變數
  MOVFF BSR,BSR_TEMP       ;儲存處理器資料重要變數

  BSF KEY_FLAG,0          ;將KEY_TEMP的bit 0設為1，代表接收過訊息
  MOVFF RCREG, RX_TEMP	  ;將RCREG值丟入RX_TEMP，以利之後讀取

  MOVFF STATUS_TEMP,STATUS
  MOVFF WREG_TEMP,WREG
  MOVFF BSR_TEMP,BSR
  RETFIE FAST
;----------------------------INITIAL SETTING----------------------
INIT_IO:
  BSF TRISA,4             ;設TRISA4為輸入
  BSF TRISB,0             ;設TRISB0為輸入
  CLRF TRISD              ;設TRISD為輸出
  CLRF LATD               ;初始讓LED全滅
  RETURN
INIT_CCP:
  BCF TRISC,2             ;設TRISC2為輸出
  MOVLW B'00101111'       ;設CCP1為PWM模式，工作週期最低2位元為00
  MOVWF CCP1CON
  MOVLW B'00000111'       ;啟動TIMER2，前除器為16倍
  MOVWF T2CON
  CLRF PR2                ;清除PR2
  RETURN
INIT_TIMER1:
  MOVLW B'10001111'       ;16位元非同步計數器模式，關閉前除器
  MOVWF T1CON             ;使用外部32768Hz震盪器並開啟TIMER1
  MOVLW (.65536-TMR_1)/.256 ;設定計時器高位元組資料
  MOVWF TMR1H
  MOVLW (.65536-TMR_1)%.256 ;設定計時器低位元組資料
  MOVWF TMR1L   	
  BCF PIR1,TMR1IF         ;清除TIMER1中斷旗標
  BSF PIE1,TMR1IE         ;開啟TIMER1中斷功能
  BSF IPR1,TMR1IP         ;設定TIMER1為高優先中斷
  RETURN
INIT_TXRX:
  MOVLW B'00100100'       ;8位元模式非同步傳輸
  MOVWF TXSTA             ;高鮑率設定，啟動傳輸功能
  MOVLW B'10010100'       ;啟動8位元資料接收功能
  MOVWF RCSTA             ;連續接收模式，停止位址偵測點
  MOVLW .64               ;設定鮑率為9600
  MOVWF SPBRG
  BCF PIR1,TXIF		  ;清除資料傳輸中斷旗標		
  BCF PIE1,TXIE		  ;停止資料傳輸中斷功能	
  BCF IPR1,RCIP	  	  ;設定資料接收為低優先中斷
  BCF PIR1,RCIF		  ;清除資料接收中斷旗標
  BSF PIE1,RCIE           ;啟動資料接收中斷
  RETURN
INIT_AD:
  MOVLW B'00001001'       ;選擇AN2通道轉換
  MOVWF ADCON0            ;啟動A/D模組
  MOVLW B'00001100'       ;設定AN0~AN2為類比入
  MOVWF ADCON1
  MOVLW B'00111111'       ;結果向左靠期並設定轉換時間為Fosc/32
  MOVWF ADCON2
  BCF PIE1,ADIE           ;關閉A/D模組中斷功能
  RETURN
INIT_SETTING:             ;清除所有暫存器                
  CLRF RX_TEMP
  CLRF KEY_FLAG
  CLRF PLAY_FLAG
  CLRF RECORD_TEMP
  CLRF ABC
  CLRF ABC_2
  CLRF ABC_3
  CLRF W_TEMP
  RETURN
INIT_ISRS:                
  BSF RCON,IPEN           ;啟動中斷優先順序
  BSF INTCON,GIEH         ;啟動高優先中斷功能，以利TIMER1中斷
  BSF INTCON,GIEL         ;啟動並優先中斷功能，以利UART中斷
  RETURN
;----------------------- 100 ms delay ---------------------- 
delay_100ms:	
  movlw	VAL_MS		 
  movwf	count_ms
loop_ms:	      
  call delay_1ms
  decfsz count_ms,f
  goto loop_ms
  return
;------------------------ 1 ms delay------------------------
delay_1ms:	 
  movlw	VAL_US		 		
  movwf	count
dec_loop:		
  call 	D_short		
  decfsz count,f		
  goto 	dec_loop		
  return
;------------------------- 5uS delay ------------------------
D_short:		
  call	D_ret			
  call	D_ret			
  nop					
  nop					
D_ret:		
  return
;------------------------- The End ------------------------
END
