const textarea = document.querySelector("section"),
speechBtn = document.querySelector("button");



let synth = speechSynthesis,
isSpeaking = true;


function textToSpeech(text){
    console.log(text);
    let utterance = new SpeechSynthesisUtterance(text);
    synth.speak(utterance);
}
speechBtn.addEventListener("click", e =>{

    e.preventDefault();
    if(textarea.value !== ""){
        if(!synth.speaking){
            textToSpeech(textarea.innerHTML);
        }

        if(isSpeaking){
            synth.resume();
            isSpeaking = false;
            speechBtn.innerText = "Pause Speech"; 
        }
        else{
            synth.pause();
            isSpeaking = true;
            speechBtn.innerText = "Resume Speech"; 
        }

        
    }
});

function voiceStop(){
    synth.cancel();
}